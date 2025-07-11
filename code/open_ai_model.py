# import the OpenAI Python library for calling the OpenAI API
from openai import OpenAI

class Open_AI:
    def __init__(self, model):
        self.model = model

    def request(self, message, exp_type):
        # Reference:
        # https://platform.openai.com/docs/api-reference/chat/create
        #
        client = OpenAI(
            api_key='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',
        )

        if exp_type == 'EXP_1':
            response = client.chat.completions.create(
                model=self.model,  # gpt-3.5-turbo / text-davinci-003
                messages=[
                    {"role": "system",
                     "content": "Given a user, as a Recommender System, please provide the top 50 recommendations."},
                    {"role": "user", "content": message}
                ],
                temperature = 0,
                max_tokens = 750,
                top_p = 1,
                frequency_penalty = 0,
                presence_penalty = 0,
            )
        else:
            response = client.chat.completions.create(
                model=self.model,  # gpt-3.5-turbo / text-davinci-003
                messages=[
                    {"role": "system", "content": "Given a user, act like a Recommender System."},
                    {"role": "user", "content": message}
                ],
                temperature=0,
                max_tokens=750,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
            )

        return response


    # def request_davinci(self, message):
    #     response = openai.Completion.create(
    #         model="text-davinci-003",
    #         prompt=message,
    #         max_tokens=900,
    #         temperature=0
    #     )
    #     return response