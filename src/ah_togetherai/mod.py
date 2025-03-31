from lib.providers.services import service
import os
import base64
from io import BytesIO
import json
from openai import AsyncOpenAI
from lib.utils.messages import concat_all_texts
import traceback

# Configure OpenAI client to use togetherai's API
client = AsyncOpenAI(
    api_key=os.environ.get("TOGETHER_API_KEY"),
    base_url="https://api.together.xyz/v1"
)

@service()
async def stream_chat(model, messages=[], context=None, num_ctx=200000, 
                     temperature=0.1, max_tokens=2000, num_gpu_layers=0):
    try:
        print("togetherai stream_chat (OpenAI compatible mode)")
        # Use env model or default
        model_name = os.environ.get("DEFAULT_LLM_MODEL", "deepseek-ai/DeepSeek-R1")
        if os.environ.get("LLM_TEMP", 0) != 0:
            temperature = float(os.environ.get("LLM_TEMP"))

        #msgs = concat_all_texts(messages)
        msgs = messages
        # Create streaming response using OpenAI compatibility layer
        stream = await client.chat.completions.create(
            model=model_name,
            messages=msgs,
            frequency_penalty=0.05,
            stream=True,
            temperature=temperature,
            max_tokens=max_tokens
        )

        print("Opened stream with model:", model_name)
        
        async def content_stream(original_stream):
            if model == "deepseek-ai/DeepSeek-R1":
                yield '[{"reasoning": "'
            done_reasoning = False

            async for chunk in original_stream:
                try:
                    if os.environ.get('AH_DEBUG') == 'True':
                        try:
                            print('\033[93m' + str(chunk) + '\033[0m', end='')
                            print('\033[92m' + str(chunk.choices[0].delta.content) + '\033[0m', end='')
                        except Exception as e:
                            pass
                    if len(chunk.choices) == 0:
                        yield ""
                        continue
                    if chunk.choices[0].delta.content.startswith("<think>"):
                        after_think = chunk.choices[0].delta.content.split("<think>")[1]
                        json_str = json.dumps(after_think)
                        without_quotes = json_str[1:-1]
                        yield without_quotes
                    elif "</think>" in chunk.choices[0].delta.content:
                        before_think = chunk.choices[0].delta.content.split("</think>")[0]
                        json_str = json.dumps(before_think)
                        without_quotes = json_str[1:-1]
                        yield without_quotes + '"}] <<CUT_HERE>>'
                        done_reasoning = True
                    elif model == "deepseek-ai/DeepSeek-R1":
                        if done_reasoning:
                            yield chunk.choices[0].delta.content
                        else:
                            json_str = json.dumps(chunk.choices[0].delta.content)
                            without_quotes = json_str[1:-1]
                            yield without_quotes
                    else:
                        yield chunk.choices[0].delta.content or ""
                except Exception as e:
                    print('togetherai (OpenAI mode) error:', e)
                    trace = traceback.format_exc()
                    print(trace)
                    yield ""

        return content_stream(stream)

    except Exception as e:
        print('togetherai (OpenAI mode) error:', e)
        #raise

@service()
async def format_image_message(pil_image, context=None):
    """Format image for togetherai using OpenAI's image format"""
    buffer = BytesIO()
    print('converting to base64')
    pil_image.save(buffer, format='PNG')
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    print('done')
    
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/png;base64,{image_base64}"
        }
    }

@service()
async def get_image_dimensions(context=None):
    """Return max supported image dimensions for togetherai"""
    return 4096, 4096, 16777216  # Max width, height, pixels
