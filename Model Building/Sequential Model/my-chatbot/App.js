import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [message, setMessage] = useState('');
  const [response, setResponse] = useState('');

  const sendMessage = async (e) => {
    e.preventDefault();
    try {
      const res = await axios.post('http://localhost:5000/predict', {
        message: message
      });
      setResponse(res.data.answer);
    } catch (error) {
      console.error("There was an error sending the message!", error);
    }
  };

  return (
    <div>
      <h1>Chatbot</h1>
      <form onSubmit={sendMessage}>
        <input
          type="text"
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          placeholder="Type your message"
        />
        <button type="submit">Send</button>
      </form>
      <p>Response: {response}</p>
    </div>
  );
}

export default App;
