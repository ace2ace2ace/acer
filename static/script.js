async function generateStory() {
    const format = document.getElementById('format').value;
    const topic = document.getElementById('topic').value;
    const length = document.getElementById('length').value;

    // Validate inputs
    if (!topic || !plot || !characters) {
        alert('Please fill in all fields');
        return;
    }

    // Show loading state
    const storyText = document.getElementById('story-text');
    storyText.innerHTML = '<p class="loading">Generating your story...</p>';

    try {
        const response = await fetch('/generate-story', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                format,
                topic,
                length
            })
        });

        const data = await response.json();
        storyText.innerHTML = data.story.split('\n').map(paragraph =>
            `<p>${paragraph}</p>`
        ).join('');

    } catch (error) {
        storyText.innerHTML = '<p class="error">Error generating story. Please try again.</p>';
        console.error('Error:', error);
    }
} 