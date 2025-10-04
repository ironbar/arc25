import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model

logger = logging.getLogger(__name__)


def get_model(model_path, torch_dtype="bfloat16", device_map='None', use_4bit_quantization=False, use_gradient_checkpointing=False):
    logger.info('Loading model...')
    log_gpu_memory()
    if use_4bit_quantization:
        logger.info('Using 4-bit quantization')
        bnb_config = BitsAndBytesConfig(
            load_in_4bit= True,
            bnb_4bit_quant_type= "nf4",
            bnb_4bit_compute_dtype= torch.float16,
            bnb_4bit_use_double_quant= True,
            llm_int8_enable_fp32_cpu_offload= True,
            llm_int8_skip_modules=['gate', 'lm_head'],
        )
    else:
        bnb_config = None
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map=_get_device_map(device_map),
        # max_memory={0: '9GB', 1: '8GB'},
        trust_remote_code=True,
        torch_dtype=_get_torch_dtype(torch_dtype), #bfloat16 is 4 times slower on Kaggle than float16, on my computer they are the same speed
        attn_implementation=_get_flash_attention_implementation(),
        )
    # print(model.hf_device_map)
    log_gpu_memory()
    if use_gradient_checkpointing: # Always disable cache when using gradient checkpointing
        model.config.use_cache = False
    if use_4bit_quantization:
        # QLoRA on Kaggle is 4 times slower than LoRA, I'm trying to disable gradient checkpointing
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=use_gradient_checkpointing)
    elif use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
    return model


def _get_device_map(device_map):
    if device_map == 'None':
        logger.info('Using None device map')
        device_map = None
    elif device_map in ['balanced', 'auto']:
        logger.info(f'Using {device_map} device map')
    else:
        raise ValueError(f'Unknown device map {device_map}')
    return device_map


def _get_torch_dtype(torch_dtype):
    if torch_dtype == 'float16':
        logger.info('Using float16 torch dtype')
        return torch.float16
    elif torch_dtype == 'bfloat16':
        logger.info('Using bfloat16 torch dtype')
        return torch.bfloat16
    else:
        raise ValueError(f'Unknown torch dtype {torch_dtype}')


def _get_flash_attention_implementation():
    try:
        import flash_attn
        attn_implementation = "flash_attention_2"
    except ImportError:
        attn_implementation = None
    logger.info(f'Using {attn_implementation} attention implementation')
    return attn_implementation


def log_gpu_memory():
    n_devices = torch.cuda.device_count()
    if n_devices == 0:
        logger.warning('No GPU is available!!!')
    for device in range(n_devices):
        logger.info(f'GPU {device} memory allocated: {torch.cuda.memory_allocated(device)/1024**3:.1f} GB, \
                    max memory allocated: {torch.cuda.max_memory_allocated(device)/1024**3:.1f} GB, \
                    GPU total memory: {torch.cuda.get_device_properties(device).total_memory/1024**3:.1f} GB')


def get_tokenizer(model_path, model, grid_encoder, pad_token='<|pad|>'):
    #TODO: delete numbers from vocabulary if necessary
    logger.info('Loading tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True)
    if tokenizer.pad_token == tokenizer.eos_token:
        if 'qwen' in model_path.lower():
            logger.info('Changing eos token to <|im_end|> for Qwen models, because it is the same as padding token <|endoftext|>')
            tokenizer.eos_token = '<|im_end|>'
        elif 'smollm' in model_path.lower():
            logger.info('Changing pad token to "<|endoftext|>" for SmolLM models, because it is the same as eos token <|im_end|>')
            tokenizer.pad_token = "<|endoftext|>"
        elif 'llama-3.1' in model_path.lower():
            logger.info('Changing pad token from <|eot_id|> to <|finetune_right_pad_id|> in the tokenizer for llama-3.1 models. Otherwise the collator does not work properly and the model does not learn to end the sequence.')
            pad_token = '<|finetune_right_pad_id|>'
            assert pad_token in tokenizer.get_vocab()
            tokenizer.pad_token = pad_token
        else:
            raise NotImplementedError(f'Changing padding token is not implemented for this model: {model_path}')
    elif 'pad_token' not in tokenizer.special_tokens_map or tokenizer.pad_token == tokenizer.eos_token:
        logger.info('Adding padding token because the tokenizer does not have one')
        assert pad_token not in tokenizer.get_vocab()
        tokenizer.add_special_tokens({'pad_token': pad_token})
        tokenizer.padding_side = 'right'
        model.resize_token_embeddings(len(tokenizer))

    assert tokenizer.pad_token != tokenizer.eos_token
    assert tokenizer.pad_token_id != tokenizer.eos_token_id
    if not 'ColorNameEncoder' in grid_encoder:
        check_tokenizer_has_unique_words_for_numbers(tokenizer)

    # ValueError: You are attempting to perform batched generation with padding_side='right' this may lead to unexpected behaviour for Flash Attention version of Qwen2. Make sure to  call `tokenizer.padding_side  = 'left'` before tokenizing the input. 
    # TODO: could this have an effect on inference?
    tokenizer.padding_side = 'left'
    return tokenizer


def check_tokenizer_has_unique_words_for_numbers(tokenizer):
    for i in range(10):
        words = get_words_with_symbol(str(i), tokenizer.get_vocab())
        if len(words) != 1:
            raise ValueError(f'Found {len(words)} words with symbol {i} in tokenizer vocabulary: {words}')
    logger.info('Tokenizer is valid, each number has a unique word in the vocabulary')


def get_words_with_symbol(symbol, vocab, skip_special_tokens=True):
    words = [word for word in vocab if symbol in word]
    if skip_special_tokens:
        words = [word for word in words if not word.startswith('<')]
    return words


def get_lora_model(model, adapter_path, r, use_rslora, use_dora, weight_initalization):
    if adapter_path is None:
        if weight_initalization == 'default': weight_initalization = True
        peft_config = LoraConfig(
            # lora_alpha: LoRA scaling factor.
            lora_alpha=64, #64,
            lora_dropout=0.1, # 0.1, althought Vaca suggested to use 0.05 for big models
            # r: the rank of the update matrices, expressed in int. Lower rank results in smaller update matrices with fewer trainable parameters.
            r=r, #16
            bias="none",
            task_type="CAUSAL_LM",
            # target_modules: The modules (for example, attention blocks) to apply the LoRA update matrices.
            target_modules= ['k_proj', 'q_proj', 'v_proj', 'o_proj'],
            use_rslora=use_rslora,
            use_dora=use_dora,
            init_lora_weights=weight_initalization # bool | Literal['gaussian', 'olora', 'pissa', 'pissa_niter_[number of iters]', 'loftq'] = True,
        )
        logger.info(f'Creating LoRA with the following config: {peft_config}')
        model = get_peft_model(model, peft_config)
    else:
        logger.info(f'Loading adapter from {adapter_path}')
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    return model