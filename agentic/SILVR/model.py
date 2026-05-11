import requests
import openai
from openai import OpenAI, AzureOpenAI
import pdb
from pprint import pprint
import time

try:
    import torch
    from transformers import (AutoTokenizer, AutoProcessor,
                              Llama4ForConditionalGeneration,
                              AutoModelForCausalLM,
                              Gemma3ForConditionalGeneration)
except ImportError:
    torch = None


def get_model(args):
    if 'dummy' in args.model:
        model = DummyModel()
    elif getattr(args, 'azure_endpoint', ''):
        model = AzureGPTModel(
            azure_endpoint=args.azure_endpoint,
            api_key=args.api_key,
            api_version=getattr(args, 'azure_api_version', '2024-12-01-preview'),
            deployment=getattr(args, 'azure_deployment', args.model),
            max_completion_tokens=getattr(args, 'max_completion_tokens', 16384),
        )
    elif len(args.endpoint) > 0:
        if 'openai' in args.endpoint.lower() or 'deepseek' in args.endpoint.lower() or 'lambda' in args.endpoint.lower():
            model = GPT(args.api_key, args.api_url, args.model)
        elif 'gemini' in args.model or 'gemma' in args.model:
            model = Gemini(args.api_key, args.model)
        else:
            raise NotImplementedError(f"Model {args.model} with endpoint {args.endpoint} not implemented")
    elif 'gpt' in args.model or 'deepseek' in args.model:
        model = GPT(args.api_key, args.api_url, args.model)
    elif 'Llama-4' in args.model:
        model = LLaMA4(args.model)
    elif 'gemma' in args.model.lower():
        model = Gemma(args.model)
    else:
        raise NotImplementedError(f"Model {args.model} not implemented")
    return model


class DummyModel():
    def __init__(self):
        super().__init__()
    def forward(self, head, prompts):
        output = {
            'response': "dummy response",
            'reasoning_content': "dummy reasoning",
            'usage': "dummy usage",
            'message': "dummy message",
        }
        return output 
    

class GPT():
    def __init__(self, api_key, api_url, model_name):
        super().__init__()
        self.api_key = api_key
        self.api_url = api_url
        self.model_name = model_name

    def forward(self, head, prompts):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        messages = []
        for i, prompt in enumerate(prompts):
            messages.append(
                {"role": "user", "content": prompt}
            )
            data = {
                "model": self.model_name,
                "messages": messages
            }
            try:
                response = requests.post(self.api_url, json=data, headers=headers)
                response.raise_for_status()  # Raise error for bad responses
                response_json = response.json()
            except Exception as e:
                print(f"Error in API. Response: {response}")
                raise
            response_text = response_json["choices"][0]["message"]["content"]
            reasoning_content = response_json["choices"][0]["message"]["reasoning_content"] if "reasoning_content" in response_json["choices"][0]["message"] else ""
            messages.append(
                {"role": "assistant", "content": response_text}
            )
            usage = dict(response_json['usage'])  # completion_tokens, prompt_tokens, total_tokens
        output = {
            'response': messages[-1]["content"],
            'reasoning_content': reasoning_content,
            'usage': usage,
            'message': messages,
        }
        return output
    

class AzureGPTModel():
    def __init__(self, azure_endpoint, api_key, api_version, deployment, max_completion_tokens=16384):
        super().__init__()
        self.client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version,
        )
        self.deployment = deployment
        self.max_completion_tokens = max_completion_tokens

    def _call_with_retry(self, messages, max_retries=5):
        for attempt in range(max_retries):
            try:
                return self.client.chat.completions.create(
                    model=self.deployment,
                    messages=messages,
                    max_completion_tokens=self.max_completion_tokens,
                )
            except openai.RateLimitError as e:
                wait = min(2 ** attempt * 2, 60)
                print(f"Rate limited (attempt {attempt+1}/{max_retries}), waiting {wait}s...")
                time.sleep(wait)
                if attempt == max_retries - 1:
                    raise

    def forward(self, head, prompts):
        messages = []
        for prompt in prompts:
            messages.append({"role": "user", "content": prompt})
            response = self._call_with_retry(messages)
            choice = response.choices[0]
            response_text = choice.message.content or ""
            reasoning_content = getattr(choice.message, 'reasoning_content', '') or ""
            messages.append({"role": "assistant", "content": response_text})
            if response.usage:
                usage = {
                    'completion_tokens': response.usage.completion_tokens,
                    'prompt_tokens': response.usage.prompt_tokens,
                    'total_tokens': response.usage.total_tokens,
                }
            else:
                usage = {}
        output = {
            'response': messages[-1]["content"],
            'reasoning_content': reasoning_content,
            'usage': usage,
            'message': messages,
        }
        return output


class Gemini():
    def __init__(self, api_key, model_name):
        super().__init__()
        self.api_key = api_key
        self.model_name = model_name

    def forward(self, head, prompts):
        from google import genai
        client = genai.Client(api_key=self.api_key)

        response = client.models.generate_content(
            model=self.model_name,
            contents=prompts[0],
        )

        output = {
            'response': response.text,
            'message': prompts,
        }
        return output


class LLaMA4():
    def __init__(self, model_name, max_new_tokens=256):
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Llama4ForConditionalGeneration.from_pretrained(
            model_name,
            attn_implementation="eager",
            device_map="auto",
            torch_dtype=torch.bfloat16,
            # cache_dir = cache_dir,
            # use_auth_token = use_auth_token
        )
        self.max_new_tokens = max_new_tokens

    def forward(self, head, prompts):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompts[0]},
                ]
            },
        ]
        
        inputs = self.processor.apply_chat_template(
            messages,
            attn_implementation="eager",
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
        )

        response = self.processor.batch_decode(outputs[:, inputs["input_ids"].shape[-1]:])[0]
        output = {
            'response': response,
            'message': messages,
        }
        return output


class Gemma():
    def __init__(self, model_name, max_new_tokens=128):
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            model_name, device_map="auto"
        ).eval()

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.max_new_tokens = max_new_tokens

    def forward(self, head, prompts):
        # messages = [
        #     {
        #         "role": "system",
        #         "content": [{"type": "text", "text": "You are a helpful assistant."}]
        #     },
        #     {
        #         "role": "user",
        #         "content": [
        #             {"type": "text", "text": prompts[0]}
        #         ]
        #     }
        # ]

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompts[0]}
                ]
            }
        ]
        
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(self.model.device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=True)
            generation = generation[0][input_len:]

        decoded = self.processor.decode(generation, skip_special_tokens=True)
        output = {
            'response': decoded,
            'message': messages,
        }
        return output