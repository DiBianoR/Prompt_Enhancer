import prompt_enhancer_torch as pe


import tkinter
if __name__ == '__main__':
    # scrape_mj_favs('./Midjourney - Image Galleries - favorites [938154212238430341] (after 2022-01-01).txt')
    # scrape_lexica(frequency=20)  # let this run for while to get quality prompts
    pe.train_autocomplete()  #  fine-tune the model for simple prompt generation
    # complete_prompt('Spiderman using a giant ruler to measure a web,')  # finish a prompt
    # extend_prompt('Spiderman using a giant ruler to measure a web', verbose=True)  # finish a prompt