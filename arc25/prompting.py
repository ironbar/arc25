from jinja2 import Template
import random
import numpy as np
from termcolor import colored

from arc25.metrics import pixel_similarity_score


def pretty_print_prompt(text, default_color='black'):
    color = default_color
    attrs = None
    print('-'*80)
    for line in text.splitlines():
        if line.startswith('<|assistant|>') or line.startswith('<|im_start|>assistant'):
            color = 'blue'
        elif line.startswith('<|user|>') or line.startswith('<|im_start|>user'):
            color = default_color
        elif line.startswith('<|system|>') or line.startswith('<|im_start|>system') or line.startswith('<|begin_of_text|><|start_header_id|>system'):
            color = 'green'
        if line.startswith('<'):
            attrs = ['bold']
        else:
            attrs = None
        print(colored(line, color, attrs=attrs))
        # llama-3
        if line.endswith('<|start_header_id|>user<|end_header_id|>'):
            color = default_color
        elif line.endswith('<|start_header_id|>assistant<|end_header_id|>'):
            color = 'blue'
        elif line.endswith('<|start_header_id|>system<|end_header_id|>'):
            color = 'green'
    print('-'*80)


def create_prompt_from_task(task, grid_encoder, tokenizer, shuffle_train_samples=True):
    messages = _create_messages_for_prompt(task, grid_encoder, tokenizer, shuffle_train_samples=shuffle_train_samples)
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, continue_final_message=True)
    return prompt


def create_refine_prompt(task, grid_encoder, tokenizer, text_prediction, train_predictions):
    messages = _create_messages_for_prompt(task, grid_encoder, tokenizer, shuffle_train_samples=False)
    messages[-1]['content'] += text_prediction
    # Add execution feedback to each train sample
    execution_outputs = []
    for i, train_sample in enumerate(task['train']):
        execution_output = train_predictions[i]
        similarity_score = pixel_similarity_score(train_sample['output'], execution_output)
        if similarity_score == 1.0:
            feedback_message = "The output grid matches the expected output grid."
            execution_outputs.append({'output': '', 'feedback_message': feedback_message})
        else:
            feedback_message = f"The output grid does not match the expected output grid.\nThe pixel similarity score is {similarity_score:.1%}."
            if np.array(train_sample['output']).shape != np.array(execution_output).shape:
                feedback_message += f"\nThe expected output grid has shape {np.array(train_sample['output']).shape}, but the generated output grid has shape {np.array(execution_output).shape}."
            else:
                feedback_message += '\nThe output grid shape is correct.'
            execution_outputs.append({'output': grid_encoder.to_text(execution_output), 'feedback_message': feedback_message})
    user_prompt = refine_prompt_template.render(execution_outputs=execution_outputs)
    messages.append({"role": "user", "content": user_prompt})
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    return prompt


def _create_messages_for_prompt(task, grid_encoder, tokenizer, shuffle_train_samples=True):
    train_samples = [{'input': grid_encoder.to_text(sample['input']), 'output': grid_encoder.to_text(sample['output'])} for sample in task['train']]
    if shuffle_train_samples:
        random.shuffle(train_samples)
    test_sample = random.choice(task['test'])
    render_kwargs = dict(train_samples=train_samples, test=grid_encoder.to_text(test_sample['input']))
    user_prompt = prompt_template.render(**render_kwargs)
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": common_prefix}]
    return messages



# https://huggingface.co/barc0/Llama-3.1-ARC-Potpourri-Induction-8B
system_prompt = """You are a world-class puzzle solver with exceptional pattern recognition skills and expertise in Python programming. Your task is to analyze puzzles and provide Python solutions."""

prompt_template_text = """Given input-output grid pairs as reference examples, carefully observe the patterns to predict the output grid for new test input. Each pair follows the same transformation rule. Grids are 2D arrays represented as strings, with cells (colors) separated by spaces and rows by newlines.
Here are the input and output grids for the reference examples:
{% for sample in train_samples %}Example {{ loop.index }}
Input:
{{ sample.input }}

Output:
{{ sample.output }}

{% endfor %}
Here is the input grid for the test example:
{{ test }}

Write a Python function `transform` that can convert any given input grid to its corresponding output grid based on the pattern observed in the reference examples.
"""

# I have verified that all responses start with this prefix
common_prefix = "Let's solve this puzzle using Python code with the common library functions. We'll first reason about the problem and then write the code to solve it. The `transform` function will take the input grid and return the output grid. Here is the Python code with the comments describing how to solve the problem:\n" #```python\nfrom common import *\n"

prompt_template = Template(prompt_template_text)

refine_prompt_template_text = """Your first solution was not correct. These are the results from executing your code:
{% for sample in execution_outputs %}Example {{ loop.index }}
Execution output:
{{ sample.output }}

{{ sample.feedback_message }}

{% endfor %}
Write a revised Python function `transform` that addresses the issues in your previous attempt and correctly transforms the input grid to the output grid based on the patterns observed in the reference examples.
"""

refine_prompt_template = Template(refine_prompt_template_text)

def parse_python_code_from_response(text):
    # Extract Python code from the text
    if '```python' not in text:
        return ''
    code = text.split('```python')[1]
    if not '```' in code:
        return ''

    code = code.split('```')[0].strip()
    return code
