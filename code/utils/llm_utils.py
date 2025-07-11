from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, pipeline

def prompt_gen(str_1, str_2, model, tokenizer, verb):

    prompt_template = f'''<s>[INST] Given two items, answer with YES or NO. YES if they represent the same item, NO if they represent different items. \
    [/INST] [INST] <The Lego Movie 2: The Second Part (2019)> and <The Lego Movie (2014)>. [/INST] NO. </s> \
    [INST] <Shrek Forever After (2010)> and <Shrek Forever After (a.k.a. Shrek: The Final Chapter) (2010)> [/INST] YES </s> \
    [INST] <The Rich Man's Wife (1996)> and <Preacher's Wife, The (1996)> [/INST] NO </s> \
    [INST] {str_1} and {str_2}[/INST]'''

    # Inference via Transformers' pipeline
    generation_params = {
        "do_sample": True,
        "temperature": 0.01,
        "top_p": 0.95,
        "top_k": 2,
        "max_new_tokens": 1,
        "repetition_penalty": 1.1
    }

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        **generation_params
    )

    pipe_output = pipe(prompt_template)[0]['generated_text']
    if verb:
        print(str_1 + " is similar to " + str_2 + ": " + pipe_output)

    if "Y" in pipe_output[len(pipe_output) - 3:]:
        return str_2

def check_item_similarity_mistral_7b(str_1, possibilities, model_name = "TheBloke/Mistral-7B-Instruct-v0.2-AWQ"):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        device_map="cuda:0"
    )

    if isinstance(possibilities, type({}.keys())):
        for str_2 in possibilities:
            prompt = ("Given two sentences, answer with YES or NO. YES if they represent the same item, NO if they represent different items. <") + str_1 + "> and <" + str_2 + ">."
            prompt_template = f'''<s>[INST] {prompt} [/INST]'''

            # Inference via Transformers' pipeline
            generation_params = {
                "do_sample": True,
                "temperature": 0.01,
                "top_p": 0.95,
                "top_k": 50,
                "max_new_tokens": 20,
                "repetition_penalty": 1.1
            }

            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                **generation_params
            )

            pipe_output = pipe(prompt_template)[0]['generated_text']
            print(str_1 + " is similar to " + possibilities[0] + ": " + pipe_output)

            prompt = pipe_output + "<s>[INST] YES or NO? "
            prompt_template = f'''{prompt} [/INST]'''

            # Inference via Transformers' pipeline
            generation_params = {
                "do_sample": True,
                "temperature": 0.01,
                "top_p": 0.95,
                "top_k": 2,
                "max_new_tokens": 1,
                "repetition_penalty": 1.1
            }

            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                **generation_params
            )

            pipe_output = pipe(prompt_template)[0]['generated_text']
            print(str_1 + " is similar to " + possibilities[0] + ": " + pipe_output[len(pipe_output) - 3:])

            if "Y" in pipe_output[len(pipe_output) - 3:]:
                return possibilities
    else:
        prompt = "Given two sentences, answer with YES or NO. YES if they represent the same item, NO if they represent different items. <" + str_1 + "> and <" + possibilities[0]+">."
        prompt_template = f'''<s>[INST] {prompt} [/INST]'''

        # Inference via Transformers' pipeline
        generation_params = {
            "do_sample": True,
            "temperature": 0.01,
            "top_p": 0.95,
            "top_k": 50,
            "max_new_tokens": 20,
            "repetition_penalty": 1.1
        }

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            **generation_params
        )

        pipe_output = pipe(prompt_template)[0]['generated_text']
        print(str_1 + " is similar to " + possibilities[0] + ": " + pipe_output)

        prompt = pipe_output + "<s>[INST] YES or NO? "
        prompt_template = f'''{prompt} [/INST]'''

        # Inference via Transformers' pipeline
        generation_params = {
            "do_sample": True,
            "temperature": 0.01,
            "top_p": 0.95,
            "top_k": 2,
            "max_new_tokens": 1,
            "repetition_penalty": 1.1
        }

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            **generation_params
        )

        pipe_output = pipe(prompt_template)[0]['generated_text']
        print(str_1 + " is similar to " + possibilities[0] + ": " + pipe_output[len(pipe_output) - 3:])

        if "Y" in pipe_output[len(pipe_output) - 3:]:
            return possibilities


def check_item_similarity_mistral_7b_v2(str_1, possibilities, verb=False):
    model_name = "TheBloke/Mistral-7B-Instruct-v0.2-AWQ"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        device_map="cuda:0"
    )

    if isinstance(possibilities, type({}.keys())) or isinstance(possibilities, list):
        for str_2 in possibilities:
            match = prompt_gen(str_1, str_2, model, tokenizer, verb)
            if match:
                return match
    else:
        match = prompt_gen(str_1, possibilities[0], model, tokenizer, verb)
        if match:
            return match

if __name__ =='__main__':
    # main()
    pass