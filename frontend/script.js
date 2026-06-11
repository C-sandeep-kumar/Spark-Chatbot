// API Configuration
const API_BASE = window.location.origin;
const CHAT_ENDPOINT = `${API_BASE}/chat`;

// DOM Elements
const messagesContainer = document.getElementById('messagesContainer');
const queryInput = document.getElementById('queryInput');
const sendBtn = document.getElementById('sendBtn');
const statusDiv = document.getElementById('statusDiv');
const sourcesModal = document.getElementById('sourcesModal');
const sourcesList = document.getElementById('sourcesList');
const closeModal = document.querySelector('.close');

// Chat history
let chatHistory = [];

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    checkHealth();
});

function setupEventListeners() {
    sendBtn.addEventListener('click', sendMessage);
    queryInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    closeModal.addEventListener('click', closeSourcesModal);
    window.addEventListener('click', (e) => {
        if (e.target === sourcesModal) {
            closeSourcesModal();
        }
    });
}

async function checkHealth() {
    try {
        const response = await fetch(`${API_BASE}/health`);
        if (!response.ok) {
            showStatus('⚠️ Service temporarily unavailable', 'error');
        }
    } catch (error) {
        showStatus('⚠️ Cannot connect to chatbot service', 'error');
    }
}

async function sendMessage() {
    const query = queryInput.value.trim();
    
    if (!query) {
        showStatus('Please enter a question', 'error');
        return;
    }
    
    if (query.length > 2000) {
        showStatus('Question is too long (max 2000 characters)', 'error');
        return;
    }
    
    // Disable input while processing
    queryInput.disabled = true;
    sendBtn.disabled = true;
    
    // Add user message to chat
    addMessage(query, 'user');
    queryInput.value = '';
    showStatus('Processing your question...', 'loading');
    
    // Show typing indicator
    const typingId = addTypingIndicator();
    
    try {
        const response = await fetch(CHAT_ENDPOINT, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: query,
                chat_history: chatHistory
            })
        });
        
        // Remove typing indicator
        removeTypingIndicator(typingId);
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to get response');
        }
        
        const data = await response.json();
        
        // Add assistant response
        addMessage(data.answer, 'assistant', data.sources, data.confidence, data.processing_time);
        
        // Update chat history
        chatHistory.push({
            role: 'user',
            content: query
        });
        chatHistory.push({
            role: 'assistant',
            content: data.answer
        });
        
        // Show success status
        const processingTime = data.processing_time.toFixed(2);
        showStatus(`✓ Response generated in ${processingTime}s (Confidence: ${(data.confidence * 100).toFixed(0)}%)`, 'success');
        
    } catch (error) {
        removeTypingIndicator(typingId);
        console.error('Error:', error);
        addMessage(`Error: ${error.message}`, 'assistant');
        showStatus(`Error: ${error.message}`, 'error');
    } finally {
        queryInput.disabled = false;
        sendBtn.disabled = false;
        queryInput.focus();
    }
}

function addMessage(content, role, sources = [], confidence = null, processingTime = null) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}-message`;
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    // Format content (handle markdown-like formatting)
    let formattedContent = content;
    formattedContent = formattedContent.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    formattedContent = formattedContent.replace(/\n/g, '<br>');
    
    const contentText = document.createElement('p');
    contentText.innerHTML = formattedContent;
    contentDiv.appendChild(contentText);
    
    // Add sources if available (for assistant messages)
    if (role === 'assistant' && sources && sources.length > 0) {
        const sourcesDiv = document.createElement('div');
        sourcesDiv.className = 'message-sources';
        
        const sourcesLink = document.createElement('a');
        sourcesLink.href = '#';
        sourcesLink.textContent = `📚 ${sources.length} source(s)`;
        sourcesLink.addEventListener('click', (e) => {
            e.preventDefault();
            showSources(sources);
        });
        sourcesDiv.appendChild(sourcesLink);
        
        if (confidence !== null) {
            const confidenceBadge = document.createElement('span');
            confidenceBadge.className = 'confidence-badge';
            confidenceBadge.textContent = `Confidence: ${(confidence * 100).toFixed(0)}%`;
            sourcesDiv.appendChild(confidenceBadge);
        }
        
        contentDiv.appendChild(sourcesDiv);
    }
    
    messageDiv.appendChild(contentDiv);
    messagesContainer.appendChild(messageDiv);
    
    // Scroll to bottom
    scrollToBottom();
}

function addTypingIndicator() {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant-message';
    messageDiv.id = 'typing-indicator';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content typing-indicator';
    
    for (let i = 0; i < 3; i++) {
        const dot = document.createElement('div');
        dot.className = 'typing-dot';
        contentDiv.appendChild(dot);
    }
    
    messageDiv.appendChild(contentDiv);
    messagesContainer.appendChild(messageDiv);
    scrollToBottom();
    
    return 'typing-indicator';
}

function removeTypingIndicator(id) {
    const element = document.getElementById(id);
    if (element) {
        element.remove();
    }
}

function showStatus(message, type) {
    statusDiv.textContent = message;
    statusDiv.className = 'status ' + type;
    
    // Auto-clear success messages after 3 seconds
    if (type === 'success') {
        setTimeout(() => {
            statusDiv.textContent = '';
            statusDiv.className = 'status';
        }, 3000);
    }
}

function showSources(sources) {
    sourcesList.innerHTML = '';
    
    sources.forEach((source, index) => {
        const sourceDiv = document.createElement('div');
        sourceDiv.className = 'source-item';
        
        const titleLink = document.createElement('a');
        titleLink.href = source.url;
        titleLink.target = '_blank';
        titleLink.textContent = source.title || `Source ${index + 1}`;
        
        const relevanceDiv = document.createElement('div');
        relevanceDiv.className = 'relevance-score';
        relevanceDiv.textContent = `Relevance Score: ${(source.relevance_score * 100).toFixed(0)}%`;
        
        sourceDiv.appendChild(titleLink);
        sourceDiv.appendChild(relevanceDiv);
        sourcesList.appendChild(sourceDiv);
    });
    
    sourcesModal.style.display = 'block';
}

function closeSourcesModal() {
    sourcesModal.style.display = 'none';
}

function scrollToBottom() {
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

// Clear chat function (optional)
function clearChat() {
    messagesContainer.innerHTML = '';
    chatHistory = [];
    queryInput.value = '';
    statusDiv.textContent = '';
    
    // Add welcome message
    const welcomeDiv = document.createElement('div');
    welcomeDiv.className = 'message assistant-message';
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.innerHTML = '<p>Chat cleared. Ask me anything about Apache Spark!</p>';
    welcomeDiv.appendChild(contentDiv);
    messagesContainer.appendChild(welcomeDiv);
}

// Keyboard shortcut: Ctrl+L to clear
document.addEventListener('keydown', (e) => {
    if (e.ctrlKey && e.key === 'l') {
        e.preventDefault();
        clearChat();
    }
});