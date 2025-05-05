## Scrap utilities
Contains functions to scrap images from Google Images (`scrap`) or extract from videos (`extract_frames.py`).

### Google Images scrpaing set-up
Create a copy of `secrets-example.json` and remove `-example` part in the name. Insert your own API keys used for Google Images scraping.

#### Obtain Custom Search Engine (CSE) ID
https://programmablesearchengine.google.com/

Create a new search engine:
- set it to search "Search the entire web";
- enable Image Search in advanced settings.

#### Obtain API Key
https://console.cloud.google.com/apis/credentials

Enable Custom Search API for the project: https://console.cloud.google.com/apis/api/customsearch.googleapis.com/

#### Run
Switch to the folder containing the script. Execute `python scrap.py`.

Modify `classes` value in the source code as needed.
