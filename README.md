# Prompt_Enhancer
 Fully AI powered promt enhancer. Allow a user to autocomplete a stable diffusion or midjourney prompt with suggested terms likely to improve image quality by choosing form successive dropdown lists.


Install(Windows):
install Anaconda and create a virtual environment from environment.yml [still need to test this]

get a copy of the tokenizer/trained weights and put them in ./GPT-Neo-125M_prompt_enh, or manually call train_prompt_enhancer.py(currently set up fo a gfx card with 24G of ram) to generate them

run prompt_enhancer_torch.py in the new environment

* requires chromedriver.exe in the base directory to scrape Lexica for additional prompts

* MidJourney prompts were scraped with DiscordChatExporter
