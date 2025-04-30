from jinja2 import Template
from termcolor import colored


def parse_grid_from_response(text, grid_encoder):
    return grid_encoder.to_grid('```grid' + text)


def create_prompt_from_task(task, grid_encoder, tokenizer,
                            is_train_prompt=True, prompt_version='code-from-examples-v3'):
    system_prompt, prompt_template, answer_template = get_prompt_templates(prompt_version)
    train_samples = [{'input': grid_encoder.to_text(grid), 'output': grid_encoder.to_text(output)} for grid, output in zip(task.inputs, task.outputs)]

    render_kwargs = dict(train_samples=train_samples)
    if prompt_version.startswith('output-from-code'):
        render_kwargs['code'] = task['code']

    user_message = prompt_template.render(**render_kwargs)
    if is_train_prompt:
        if prompt_version.startswith('code-from-examples'):
            output = '```python\n' + task.code + '\n```'
        else:
            raise ValueError(f'Unknown prompt version {prompt_version}')
    else:
        if prompt_version.startswith('code-from-examples'):
            output = '```python\n'
        else:
            output = '```grid'
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": answer_template.render(output=output)}]
    prompt = tokenizer.apply_chat_template(messages,
                                            tokenize=False,
                                            add_generation_prompt=False)
    if not is_train_prompt:
        prompt = remove_assistant_ending(prompt)
    return prompt



def remove_assistant_ending(text):
    """
phi-3

```
<|assistant|>
### Output
```grid
<|end|>
<|endoftext|>
```

llama 3.1

```
<|eot_id|><|start_header_id|>assistant<|end_header_id|>

### Output
```grid<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```
    """
    if '<|eot_id|>' in text: # llama
        split_text = '<|eot_id|>'
    elif '<|im_end|>' in text: # qwen
        split_text = '<|im_end|>'
    elif '<|end|>' in text:
        split_text = '<|end|>' # phi-3
    else:
        NotImplementedError('Unknown chat template')
    return split_text.join(text.split(split_text)[:-1])


def pretty_print_smallest_prompt(prompts):
    smallest_prompt = sorted(prompts, key=lambda x: len(x))[0]
    print('\n\nSmaller prompt:')
    pretty_print_prompt(smallest_prompt)
    print('\n\n')


def pretty_print_prompt(text, default_color='black'):
    color = default_color
    attrs = None
    print('-'*80)
    for line in text.splitlines():
        if line.startswith('<|assistant|>') or line.startswith('<|im_start|>assistant'):
            color = 'blue'
        elif line.startswith('<|user|>') or line.startswith('<|im_start|>user'):
            color = default_color
        elif line.startswith('<|system|>') or line.startswith('<|im_start|>system'):
            color = 'green'
        if line.startswith('<'):
            attrs = ['bold']
        else:
            attrs = None
        print(colored(line, color, attrs=attrs))
    print('-'*80)


def get_prompt_templates(prompt_version):
    """
    Given a string defining the prompt version returns the system, prompt and answer templates.

    This are the planned prompt versions to release:

    output-from-examples
    input-from-inputs
    output-from-outputs
    code-from-examples
    output-from-code
    input-from-code
    code-from-inputs
    """
    if prompt_version == 'code-from-examples-v3':
        return system_prompt_v1, prompt_template_code_from_examples_v3, answer_template_code_from_examples_v2
    else:
        raise ValueError(f'Unknown prompt version {prompt_version}')


# v1 reduce the number of prompt tokens from 292 to 88, freeing 200 tokens
system_prompt_v1 = "You are a helpful assistant."

answer_template_code_from_examples_v2 = Template("""## Code

This is the Python function that implements the transformation logic:

{{ output }}""")

prompt_template_code_from_examples_v3 = Template("""You are tasked with solving a transformation problem from the Abstraction and Reasoning Challenge (ARC).
The goal is to generate a Python function called `task` that receives a 2D numpy array, `img`, and transforms it to match the desired output.

Below are several input-output examples that illustrate the transformation. Your function should generalize the pattern from these examples to solve any input following the same logic.

## Key Priors:

- **Objectness**: Consider the grid as containing objects (groups of connected cells) rather than just individual pixels.
- **Goal-Directed**: The transformation should achieve a specific goal, such as creating symmetry or changing the color of specific objects.
- **Numbers & Counting**: Keep track of the number of objects, sizes, and their relative positions.
- **Geometry & Topology**: Use spatial relationships such as adjacency, enclosure, or symmetry.

Carefully analyze the examples and find the underlying transformation logic.

## Examples
{% for sample in train_samples %}
### Example {{ loop.index }}

#### Input

{{ sample.input }}

#### Output

{{ sample.output }}
{% endfor %}
""")
