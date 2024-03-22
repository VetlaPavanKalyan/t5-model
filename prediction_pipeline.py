from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import pipeline
from pathlib import Path
import lz4.frame
import re

cache = {}

class PredictionPipeline:
    def __init__(self, model_dir=Path('./trained_model'), tokenizer_dir=Path('./trained_tokenizer')):
        self.model = T5ForConditionalGeneration.from_pretrained(model_dir)
        self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_dir)

    @staticmethod
    def preprocess_text(text: str) -> str:
        # Remove any URLs
        text = re.sub(r'http\S+', '', text)
        # Replace double quotes with single quotes
        text = text.replace('"', "'")
        # Remove special characters
        text = re.sub(r'[^\w\s]', '', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def compress_text(self, text: str) -> bytes:
        # Compress text using lz4
        return lz4.frame.compress(text.encode())

    def decompress_text(self, compressed_text: bytes) -> str:
        # Decompress text using lz4
        return lz4.frame.decompress(compressed_text).decode()

    def main(self, text: str) -> str:
        text = self.preprocess_text(text)

        # Check if summary is already cached
        if text in cache:
            print("Already predicted and cached:")
            return self.decompress_text(cache[text])

        gen_kwargs = {"length_penalty": 0.8, "num_beams": 8, "max_length": 128}

        pipe = pipeline("summarization", model=self.model, tokenizer=self.tokenizer)

        print("Input:")
        print(text)

        output = pipe(text, **gen_kwargs)[0]["summary_text"]
        print("\nModel Summary:")
        print(output)

        # Compress and cache the summary
        compressed_summary = self.compress_text(output)
        cache[text] = compressed_summary

        return output
