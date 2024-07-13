from transformers import pipeline


def summarize_text(reviews):

    # Combine reviews into a single string
    combined_reviews = " ".join(reviews)

    # Load the summarization pipeline
    summarizer = pipeline("summarization")

    # Summarize the combined reviews
    summary = summarizer(combined_reviews, max_length=50, min_length=25, do_sample=False)

    # Display the summary
    # print("Summary of reviews:")
    # print(summary[0]['summary_text'])
    return summary[0]['summary_text']
