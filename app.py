
from flask import Flask, request, render_template
import plotly.graph_objects as go
import numpy as np
from run import run_inference, idx_to_class, max_len

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = request.form['text']
        if not text:
            return render_template('index.html', error="Please enter some text.")
        
        # Get sequence for words
        words = text.split()[::-1]
        if len(words) > max_len:
            words = words[:max_len]
        
        # Run prediction
        pred, attention = run_inference(text)
        attention = attention[:len(words)]  # Trim to word length
        attention = attention[::-1] # Reverse the attention weights
        attention = attention / np.sum(attention) # Normalize the attention weights
        
        # Create prediction bar chart
        classes = [idx_to_class[i] for i in range(len(idx_to_class))]
        fig_pred = go.Figure(go.Bar(
            y=classes,
            x=pred,
            orientation='h',
            marker_color='skyblue'
        ))
        fig_pred.update_layout(
            title='Bias Prediction Probabilities',
            xaxis_title='Probability',
            yaxis_title='Bias Class',
            xaxis_range=[0, 1],
            height=300,
            margin=dict(l=100, r=20, t=50, b=50)
        )
        
        # Create attention bar chart
        fig_attn = go.Figure(go.Bar(
            y=words,
            x=attention,
            orientation='h',
            marker_color='lightgreen'
        ))
        fig_attn.update_layout(
            title='Attention Weights for Words',
            xaxis_title='Attention Weight',
            yaxis_title='Words',
            xaxis_range=[0, 1],
            height=400 + (len(words) * 20),
            margin=dict(l=150, r=20, t=50, b=50)
        )
        
        # Convert to HTML
        pred_plot = fig_pred.to_html(full_html=False, include_plotlyjs='cdn')
        attn_plot = fig_attn.to_html(full_html=False, include_plotlyjs=False)
        
        return render_template('index.html', text=text, pred_plot=pred_plot, attn_plot=attn_plot)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
