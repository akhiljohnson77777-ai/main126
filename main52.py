import requests
from colorama import Fore, Style, init
from config import HF_API_KEY

init(autoreset=True)

CAPTION_MODEL = "Salesforce/blip-image-captioning-base"
GPT2_MODEL = "gpt2"

CAPTION_API = f"https://router.huggingface.co/hf-inference/models/{CAPTION_MODEL}"
GPT2_API = f"https://router.huggingface.co/hf-inference/models/{GPT2_MODEL}"

HEADERS = {
    "Authorization": f"Bearer {HF_API_KEY}"
}


def generate_caption(image_path):
    with open(image_path, "rb") as img:
        response = requests.post(
            CAPTION_API,
            headers=HEADERS,
            data=img.read(),
            timeout=120
        )

    if response.status_code != 200:
        raise Exception(response.text)

    result = response.json()

    if isinstance(result, list):
        return result[0]["generated_text"]

    raise Exception("Caption generation failed")


def expand_caption(caption):
    prompt = (
        f"Write a detailed description of about 30 words for this caption: "
        f"{caption}"
    )

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 40,
            "temperature": 0.7
        }
    }

    response = requests.post(
        GPT2_API,
        headers=HEADERS,
        json=payload,
        timeout=120
    )

    if response.status_code != 200:
        raise Exception(response.text)

    result = response.json()

    if isinstance(result, list):
        return result[0]["generated_text"]

    raise Exception("Description generation failed")


def main():
    image_path = input("Enter image path: ").strip()

    try:
        print(Fore.CYAN + "\nGenerating caption...\n")

        caption = generate_caption(image_path)

        print(Fore.GREEN + Style.BRIGHT + "Caption:")
        print(Fore.GREEN + caption)

        choice = input(
            "\nGenerate a longer 30-word description? (yes/no): "
        ).strip().lower()

        if choice == "yes":
            print(Fore.CYAN + "\nGenerating description...\n")

            description = expand_caption(caption)

            print(Fore.GREEN + Style.BRIGHT + "Description:")
            print(Fore.GREEN + description)

    except Exception as e:
        print(Fore.RED + Style.BRIGHT + "Error:")
        print(Fore.RED + str(e))


if __name__ == "__main__":
    main()