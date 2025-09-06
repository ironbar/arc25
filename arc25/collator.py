import logging
from trl import DataCollatorForCompletionOnlyLM

logger = logging.getLogger(__name__)


def get_data_collator(tokenizer):
    if '<|start_header_id|>' in tokenizer.chat_template and '<|end_header_id|>' in tokenizer.chat_template:
        logger.info('Using llama template for collator')
        data_collator = DataCollatorForCompletionOnlyLM(
            tokenizer=tokenizer,
            instruction_template='<|start_header_id|>user<|end_header_id|>',
            response_template='<|start_header_id|>assistant<|end_header_id|>',
        )
    elif '<|im_start|>' in tokenizer.chat_template:
        logger.info('Using SmolLM\Qwen template for collator')
        data_collator = DataCollatorForCompletionOnlyLM(
            tokenizer=tokenizer,
            instruction_template='<|im_start|>user',
            response_template='<|im_start|>assistant',
        )
    elif '<|user|>' in tokenizer.chat_template and '<|assistant|>' in tokenizer.chat_template:
        logger.info('Using Phi-3 template for collator')
        data_collator = DataCollatorForCompletionOnlyLM(
            tokenizer=tokenizer,
            instruction_template='<|user|>',
            response_template='<|assistant|>'
        )
    else:
        raise NotImplementedError(f'Tokenizer chat template not recognized: {tokenizer.chat_template}')
    return data_collator
