# https://rapidapi.com/sparior/api/yahoo-finance15/

import json
import requests

Stock = "amd"

url = "https://yahoo-finance127.p.rapidapi.com/news/" + Stock

headers = {
    "X-RapidAPI-Key": "b498aede6fmshe8650274d3bd257p117e39jsnfdfedc94febe",
    "X-RapidAPI-Host": "yahoo-finance127.p.rapidapi.com"
}

response = requests.get(url, headers=headers)


if response.status_code == 200:
    try:
        # Parse the response text as JSON
        articles = json.loads(response.text)

        # Extract URLs from the JSON response
        urls = [articles[key]['link'] for key in articles]

        # Write the URLs to a file
        with open("news.txt", "w") as file:
            for url in urls:
                file.write(url + '\n')
        print("URLs saved to news.txt")

    except json.JSONDecodeError as e:
        print("Failed to decode JSON:", e)
else:
    print("Failed to fetch news data. Status code:", response.status_code)