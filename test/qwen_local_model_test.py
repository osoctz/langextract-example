import os

import langextract as lx
import textwrap

def add_language_markers(text, lang_code='zh'):
    """为纯文本添加语言标记"""
    return f"<lang:{lang_code}>{text}</lang:{lang_code}>"

if __name__ == '__main__':
    prompt = textwrap.dedent("""\
        Extract characters, emotions, and relationships in order of appearance.
        Use exact text for extractions. Do not paraphrase or overlap entities.
        Provide meaningful attributes for each entity to add context.""")

    # 2. Provide a high-quality example to guide the model
    examples = [
        lx.data.ExampleData(
            text="ROMEO. But soft! What light through yonder window breaks? It is the east, and Juliet is the sun.",
            extractions=[
                lx.data.Extraction(
                    extraction_class="character",
                    extraction_text="ROMEO",
                    attributes={"emotional_state": "wonder"}
                ),
                lx.data.Extraction(
                    extraction_class="emotion",
                    extraction_text="But soft!",
                    attributes={"feeling": "gentle awe"}
                ),
                lx.data.Extraction(
                    extraction_class="relationship",
                    extraction_text="Juliet is the sun",
                    attributes={"type": "metaphor"}
                ),
            ]
        )
    ]
    # The input text to be processed
    input_text = """
    Lady Juliet gazed longingly at the stars, her heart aching for Romeo
    """

    marked_text =add_language_markers(input_text)
    config = lx.factory.ModelConfig(
        model_id="vllm:http://127.0.0.1:9001/v1",
        provider="VLLMLanguageModel",
        provider_kwargs=dict(
            temperature=0.7,
            max_tokens=1024,
            # Server connection settings
            timeout=60.0,
        ),
    )

    model = lx.factory.create_model(config)

    # Run the extraction
    result = lx.extract(
        text_or_documents=marked_text,
        prompt_description=prompt,
        examples=examples,
        model=model,
    )

    print("==========")
    print(result)