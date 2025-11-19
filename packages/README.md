# DomynClaimAlign

This is a repository for the package.

What is needed?

A system that produces answer to a query.
 - Need: query, answer

If the system is simple LLM chat, means you are trying to trace directly from the training data. Then the following are required:
 - Base directory path containing the training data in jsonl format. It can have multiple files.
 - Infinigram index directory path containing the infinirgam index. You have to create the index yourself.
 - Unigram path containing a json file with unigram probabilities (if you have computed it yourself). If you have not computed it, use 
 ```from domynclaimalign.utils.calculate_and_save_unigram_probability import calculate_and_save_unigram_probability
 calculate_and_save_unigram_probability(data_dir, save_dir, tokenizer)```
 Use the tokenizer based on the model that you used for creating the infinigram index.
 - Model, which was used for infinigram idnex creation (allenai/OLMo-7B, gpt2)

 
