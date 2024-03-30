import io

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from PIL import Image
from wordcloud import WordCloud

# Combine the questions and answers into separate texts
questions_text = " ".join(result_df["question"])
answers_text = " ".join(result_df["answer"])

# Create separate WordCloud objects for questions and answers
wordcloud_questions = WordCloud(
    width=400, height=200, background_color="white").generate(questions_text)
wordcloud_answers = WordCloud(
    width=400, height=200, background_color="white").generate(answers_text)

# Convert the word clouds to images
image_questions = wordcloud_questions.to_image()
image_answers = wordcloud_answers.to_image()

# Merge the images horizontally
merged_image = Image.new("RGB", (image_questions.width + image_answers.width,
                         max(image_questions.height, image_answers.height)))
merged_image.paste(image_questions, (0, 0))
merged_image.paste(image_answers, (image_questions.width, 0))

# Convert the merged image to a Plotly figure
image_fig = go.Figure(go.Image(z=merged_image))

# Update the layout to hide axis and labels
image_fig.update_layout(
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
)

# Display the Plotly figure
image_fig.show()
