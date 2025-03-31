import plotly.graph_objects as go

def create_classification_chart(results):
    categories = list(results.keys())
    scores = list(results.values())
    
    fig = go.Figure(go.Bar(
        x=categories,
        y=scores,
        marker_color='rgba(55, 128, 191, 0.7)',
        text=[f"{score:.2f}" for score in scores],
        textposition='auto'
    ))
    
    fig.update_layout(
        title='Classification Confidence Scores',
        xaxis_title='Categories',
        yaxis_title='Confidence',
        yaxis_range=[0, 1],
        hovermode='x',
        template='plotly_white'
    )
    
    return fig.to_html(full_html=False, include_plotlyjs='cdn')

def create_similarity_chart(results):
    images = [os.path.basename(r['path']) for r in results]
    scores = [r['similarity'] for r in results]
    
    fig = go.Figure(go.Bar(
        x=images,
        y=scores,
        marker_color='rgba(75, 192, 192, 0.7)',
        text=[f"{score:.2f}" for score in scores],
        textposition='auto'
    ))
    
    fig.update_layout(
        title='Search Results Similarity Scores',
        xaxis_title='Images',
        yaxis_title='Similarity',
        yaxis_range=[0, 1],
        hovermode='x',
        template='plotly_white'
    )
    
    return fig.to_html(full_html=False, include_plotlyjs='cdn')