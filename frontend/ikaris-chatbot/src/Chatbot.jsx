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

    try {
      // Replace with your actual API endpoint
      const response = await fetch('https://api.anthropic.com/v1/messages', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: 'claude-sonnet-4-20250514',
          max_tokens: 1000,
          messages: [...messages, userMessage].map(msg => ({
            role: msg.role,
            content: msg.content
          }))
        })
      });

      const data = await response.json();
      const assistantMessage = {
        role: 'assistant',
        content: data.content[0].text
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Error:', error);
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: 'Sorry, I encountered an error. Please try again.'
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
                <div className="loading-dots">
                  <div className="dot dot1"></div>
                  <div className="dot dot2"></div>
                  <div className="dot dot3"></div>
                </div>
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