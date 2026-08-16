import arxiv

client = arxiv.Client()
search = arxiv.Search(query="large language models", max_results=2)

for result in client.results(search):
    print("Title:", result.title)
    print("Authors:", [a.name for a in result.authors])
    print("Summary:", result.summary)
    print()