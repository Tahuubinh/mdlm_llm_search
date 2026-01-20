from transformers import GPT2TokenizerFast
tok = GPT2TokenizerFast.from_pretrained("gpt2")
text = "Cryptosporidium is a parasite that can be found in soil, water, food, and water. This parasite is found in many forms, including corn, rice, green beans, seeds, and seeds, feed, bedding, and the protein used for raising and feeding sheep. With the concentration of these parasites, all foreign food can be transverted your food, water you, and make you extremely unhappy and extremely hungry, all you get something you don’t normally, all foreign food"
print(len(tok.encode(text)))