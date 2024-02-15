import requests
import time

class Completion:
    # Your existing model definition...

    @staticmethod
    def create(prompt='hello world', messages=[]):
        try:
            response = requests.post('https://www.t3nsor.tech/api/chat', json={
                **Completion.model,
                'messages': messages,
                'key': '',
                'prompt': prompt
            }, headers=headers)

            # Check if the response status code is successful
            response.raise_for_status()

            # Attempt to parse JSON only if the response is successful
            data = response.json()

            # Your existing response handling logic...
            return T3nsorResponse({
                # Your existing response parsing...
            })

        except requests.exceptions.HTTPError as http_err:
            print(f'HTTP error occurred: {http_err}')  # Python 3.6+
        except requests.exceptions.RequestException as err:
            print(f'Request error occurred: {err}')  # Python 3.6+
        except requests.exceptions.JSONDecodeError as json_err:
            print(f'Error decoding JSON: {json_err}')
            print('Response content:', response.text)

# Remember to define your T3nsorResponse, headers, and other required classes and variables as per your original code

if __name__ == "__main__":
    prompt = "hello world"
    messages = []
    response = create_t3nsor_response(prompt, messages)
    print(response)
