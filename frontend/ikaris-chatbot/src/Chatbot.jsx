import React, { useState, useRef, useEffect } from 'react';
import { Send, Bot, User } from 'lucide-react';
import './Chatbot.css';
import wingsLogo from './wings-logo.png';

const Chatbot = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [hasMessages, setHasMessages] = useState(false); // Start as false so animation triggers
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async () => {
    if (!input.trim() || isLoading) return;

    // Mark that user has started chatting
    setHasMessages(true);

    const userMessage = { role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    // Track start time for minimum loading display
    const startTime = Date.now();

    try {
      // Replace with your actual API endpoint
      const response = await fetch('http://localhost:3001/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: input })
      });

      const data = await response.json();
      const assistantMessage = {
          role: 'assistant',
          content: data.answer || 'Sorry, no answer returned.'
      };

      // Ensure loading indicator shows for at least 3000ms (3 seconds) to prevent flicker
      const elapsedTime = Date.now() - startTime;
      const minDisplayTime = 3000;
      
      if (elapsedTime < minDisplayTime) {
        await new Promise(resolve => setTimeout(resolve, minDisplayTime - elapsedTime));
      }

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Error:', error);
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: '⚠️ Sorry, I encountered an error connecting to the backend. Please ensure the Flask server is running on port 3001.\n\nError: ' + error.message
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <div className="chatbot-container">
      {/* Header */}
      <div className="chatbot-header">
        <div className="powered-by">Powered by NemoTron and CBRE.</div>
        <div className="header-content">
          <div className="bot-icon-wrapper">
            <img src={wingsLogo} alt="Wings Logo" className="wings-logo" />
          </div>
          <div>
            <h1 className="header-title">IKARIS</h1>
          </div>
        </div>
      </div>

      {/* Input Area */}
      <div className={`input-section ${hasMessages ? 'input-bottom' : 'input-center'}`}>
        <div className="input-wrapper">
          <div className="input-container">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Type your message..."
              className="input-textarea"
              rows="1"
            />
            <button
              onClick={handleSendMessage}
              disabled={!input.trim() || isLoading}
              className="send-button"
            >
              <Send className="send-icon" />
            </button>
          </div>
        </div>
      </div>

      {/* Messages Container */}
      <div className={`messages-container ${hasMessages ? 'messages-expanded' : 'messages-collapsed'}`}>
        <div className="messages-wrapper">
          {messages.map((message, index) => (
            <div
              key={index}
              className={`message-row ${message.role === 'user' ? 'user-message' : 'assistant-message'}`}
            >
              {/* Avatar */}
              <div className={`avatar ${message.role === 'user' ? 'user-avatar' : 'assistant-avatar'}`}>
                {message.role === 'user' ? (
                  <User className="avatar-icon" />
                ) : (
                  <Bot className="avatar-icon" />
                )}
              </div>

              {/* Message Bubble */}
              <div className={`message-bubble ${message.role === 'user' ? 'user-bubble' : 'assistant-bubble'}`}>
                <p className="message-text">
                  {message.content}
                </p>
              </div>
            </div>
          ))}

          {/* Loading Indicator */}
          {isLoading && (
            <div className="message-row assistant-message">
              <div className="avatar assistant-avatar">
                <Bot className="avatar-icon" />
              </div>
              <div className="message-bubble assistant-bubble">
                <div className="loading-text">...</div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </div>
    </div>
  );
};

export default Chatbot;