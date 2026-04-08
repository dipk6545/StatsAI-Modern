import { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import { Chart as ReactChartJS } from 'react-chartjs-2';
import 'chart.js/auto';

import { Paperclip, Send, FileText, Trash2, Settings, Sparkles, Check, Loader2, Play, Cpu, X, BookOpen, Menu, Activity } from 'lucide-react';
import styles from './App.module.css';

// Custom Chart.js Renderer
const ReactChartComponent = ({ config }) => {
  try {
    const parsedConfig = typeof config === 'string' ? JSON.parse(config) : config;
    const options = {
      responsive: true,
      maintainAspectRatio: false,
      color: '#ffffff',
      scales: parsedConfig.options?.scales || { x: { ticks: { color: '#ccc' } }, y: { ticks: { color: '#ccc' } } },
      plugins: {
        legend: { labels: { color: '#fff' } }
      },
      ...parsedConfig.options
    };
    return (
      <div className={styles.plotlyWrapper} style={{ backgroundColor: 'transparent', padding: '10px', height: '300px' }}>
        <ReactChartJS type={parsedConfig.type || 'bar'} data={parsedConfig.data || {datasets:[]}} options={options} />
      </div>
    );
  } catch (err) {
    return <div className={styles.chartError} style={{color: 'red'}}>Invalid Chart Config: {err.message}</div>;
  }
};

function App() {
  const [file, setFile] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [appStatus, setAppStatus] = useState('idle');
  const [instruction, setInstruction] = useState('');
  const [workflowSteps, setWorkflowSteps] = useState([]);
  const [dragging, setDragging] = useState(false);
  
  const [engineMode, setEngineMode] = useState('single'); // 'single' | 'multi'
  const [explanationModal, setExplanationModal] = useState(null); 
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [isLogsOpen, setIsLogsOpen] = useState(false);  
  const [specialization, setSpecialization] = useState(null); // 'statistics' | 'probability' | 'data_science' | 'research'  
  const messagesEndRef = useRef(null);
  
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, appStatus]);

  // Render Markdown natively, but intercept <react_chart> explicitly for Chart.js
  const renderMessageContent = (text) => {
    const parts = text.split(/(<react_chart>[\s\S]*?<\/react_chart>)/g);
    return parts.map((part, i) => {
      if (part.startsWith('<react_chart>')) {
        const configStr = part.replace('<react_chart>', '').replace('</react_chart>', '').trim();
        return <ReactChartComponent key={i} config={configStr} />;
      }
      return <ReactMarkdown key={i}>{part}</ReactMarkdown>;
    });
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setDragging(true);
  };
  
  const handleDragLeave = (e) => {
    e.preventDefault();
    setDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const droppedFile = e.dataTransfer.files[0];
      if (droppedFile.type === 'application/pdf') {
        setFile(droppedFile);
      } else {
        alert("Only PDF files are supported format.");
      }
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() && !file) return;

    const originalText = input || 'Please analyze the attached document.';
    const userMessage = { role: 'user', text: originalText, originalText: originalText, isOptimized: false };
    
    const messageIndexToUpdate = messages.length; 
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setAppStatus('optimizing');
    
    setWorkflowSteps([
      { id: '1', title: 'Received User Prompt', details: `"${originalText}"\nChecking for custom optimization rules...`, status: 'completed' },
      { id: '2', title: 'Optimizing Prompt Context', details: 'Running optimization pipeline...', status: 'running' }
    ]);

    try {
      let finalPrompt = originalText;
      let optimizationSuccess = false;
      
      if (instruction.trim().length > 0) {
          try {
              const optimizeRes = await axios.post(`http://${window.location.hostname}:3001/api/optimize-prompt`, {
                  message: originalText,
                  instruction: instruction
              });
              
              if (optimizeRes.data.optimizedPrompt && optimizeRes.data.optimizedPrompt.trim() !== '') {
                  finalPrompt = optimizeRes.data.optimizedPrompt;
                  optimizationSuccess = true;
                  
                  setMessages(prev => {
                      const newMsgs = [...prev];
                      newMsgs[messageIndexToUpdate] = { 
                        ...newMsgs[messageIndexToUpdate], 
                        text: finalPrompt,
                        isOptimized: true 
                      };
                      return newMsgs;
                  });
              }
          } catch (optErr) {
              console.error("Optimization failed, falling back to original prompt", optErr);
          }
      }

      setAppStatus('generating');
      
      const userWantsExplanation = /(explain|what is the meaning of|what is this|explain the above|what the hell is this)/i.test(originalText);

      setWorkflowSteps(prev => [
        prev[0],
        { 
          id: '2', 
          title: 'Optimization Complete', 
          details: optimizationSuccess ? `Rewritten:\n${finalPrompt}` : 'Optimization bypassed or failed.', 
          status: 'completed' 
        },
        { 
          id: '3', 
          title: engineMode === 'multi' ? (userWantsExplanation ? 'Multi-Model Parallel Sync' : 'Single-Model Pipeline') : 'Single Model Engine', 
          details: engineMode === 'multi' 
             ? `Groq computing core answer...${userWantsExplanation ? '\nCerebras formulating explanation...' : '\n(Explanation engine bypassed)'}`
             : 'Groq executing analysis...', 
          status: 'running' 
        }
      ]);

      const formData = new FormData();
      formData.append('message', finalPrompt);
      formData.append('mode', engineMode);
      formData.append('specialization', specialization || 'statistics');
      formData.append('history', JSON.stringify(messages)); 
      if (file) {
        formData.append('document', file);
      }

      const response = await axios.post(`http://${window.location.hostname}:3001/api/chat`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      
      let rawAIResponse = response.data.reply;
      
      const explanationRegex = /<explanation>([\s\S]*?)<\/explanation>/ig;
      let match;
      let extractedExplanation = '';
      
      while ((match = explanationRegex.exec(rawAIResponse)) !== null) {
          extractedExplanation += match[1].trim() + '\n\n';
      }
      
      if (extractedExplanation) {
          setExplanationModal(extractedExplanation);
          rawAIResponse = rawAIResponse.replace(explanationRegex, '').trim();
          
          if (!rawAIResponse) {
             rawAIResponse = `*A detailed conceptual explanation was generated and opened in an external floating window for you to read.*`;
          } else {
             rawAIResponse += `\n\n*(Extended conceptual explanation opened in floating window)*`;
          }
      }
      
      const aiMessage = { role: 'model', text: rawAIResponse };
      setMessages(prev => [...prev, aiMessage]);
      if (file) setFile(null); 
      
      setWorkflowSteps(prev => [
        prev[0], prev[1],
        { id: '3', title: 'Calculations Completed', details: 'Rendered to Markdown stream.', status: 'completed' },
        { id: '4', title: 'Pipeline Done', details: 'Ready for next input.', status: 'completed' },
      ]);
      
    } catch (error) {
      console.error('Error sending message:', error);
      setMessages(prev => [...prev, { role: 'model', text: '**Error:** ' + (error.response?.data?.error || error.message || 'Failed to connect.') }]);
      
      setWorkflowSteps(prev => [
        ...prev.filter(p => p.status === 'completed'),
        { id: 'err', title: 'Pipeline Error', details: error.message, status: 'error' }
      ]);
    } finally {
      setAppStatus('idle');
    }
  };

  const getStepIcon = (status) => {
    switch (status) {
      case 'completed': return <div className={`${styles.stepIcon} ${styles.stepIconDone}`}><Check size={14} /></div>;
      case 'running': return <div className={`${styles.stepIcon} ${styles.stepIconActive}`}><Loader2 size={14} className={styles.loadingText} /></div>;
      default: return <div className={styles.stepIcon}><Play size={10} style={{ marginLeft: '2px' }}/></div>;
    }
  };

  return (
    <>
      {/* FLOATING EXPLANATION WINDOW */}
      {explanationModal && (
        <div className={styles.modalOverlay} onClick={() => setExplanationModal(null)}>
           <div className={styles.modalContent} onClick={e => e.stopPropagation()}>
               <button className={styles.modalClose} onClick={() => setExplanationModal(null)}>
                  <X size={24} />
               </button>
               <h2><BookOpen size={20} /> Conceptual Explanation</h2>
               <div className="markdown-body" style={{ background: 'transparent' }}>
                  {renderMessageContent(explanationModal)}
               </div>
           </div>
        </div>
      )}

      <div className={styles.appContainer}>
        {/* MOBILE HEADER */}
        <header className={styles.mobileHeader}>
          <button className={styles.iconButton} onClick={() => { setIsSidebarOpen(!isSidebarOpen); setIsLogsOpen(false); }}>
            <Menu size={24} />
          </button>
          <div className={styles.mobileLogo}>
            <div className={styles.logoIconSmall}></div>
            <span>Stats AI</span>
          </div>
          <button className={styles.iconButton} onClick={() => { setIsLogsOpen(!isLogsOpen); setIsSidebarOpen(false); }}>
            <Activity size={24} />
          </button>
        </header>

        {/* OVERLAYS FOR MOBILE DRAWERS */}
        {(isSidebarOpen || isLogsOpen) && (
          <div className={styles.drawerOverlay} onClick={() => { setIsSidebarOpen(false); setIsLogsOpen(false); }} />
        )}

        {/* LEFT SIDEBAR */}
        <div className={`${styles.sidebar} ${isSidebarOpen ? styles.sidebarOpen : ''}`}>
          <div className={styles.header}>
            <div className={styles.logo}>
              <div className={styles.logoIcon}></div>
              <h1>PDF Summarizer</h1>
            </div>
            <p>Statistics & Probability Tool</p>
          </div>

          <div className={styles.sidebarSection}>
            <h3><Cpu size={14} style={{ marginRight: '6px', verticalAlign: 'middle' }}/> AI Engine Status</h3>
            <div style={{ display: 'flex', gap: '8px', marginBottom: '1.5rem', marginTop: '0.5rem' }}>
                <button 
                  onClick={() => setEngineMode('single')}
                  style={{
                    flex: 1, padding: '8px', borderRadius: '8px', fontSize: '0.8rem', cursor: 'pointer', border: 'none',
                    background: engineMode === 'single' ? 'var(--accent-color)' : 'rgba(255,255,255,0.05)',
                    color: engineMode === 'single' ? 'white' : 'var(--text-secondary)'
                  }}
                >
                  Single Model<br/>(Groq)
                </button>
                <button 
                  onClick={() => setEngineMode('multi')}
                  style={{
                    flex: 1, padding: '8px', borderRadius: '8px', fontSize: '0.8rem', cursor: 'pointer', border: 'none',
                    background: engineMode === 'multi' ? 'var(--accent-color)' : 'rgba(255,255,255,0.05)',
                    color: engineMode === 'multi' ? 'white' : 'var(--text-secondary)'
                  }}
                >
                  Multi-Model<br/>(+Cerebras Engine)
                </button>
            </div>
          </div>

          <div className={styles.sidebarSection}>
            <h3><Settings size={14} style={{ marginRight: '6px', verticalAlign: 'middle' }}/> Optimization Instruction</h3>
            <p style={{ fontSize: '0.8rem', opacity: 0.7, marginBottom: '0.5rem' }}>
              Override default prompt behaviors instantly before dispatch.
            </p>
            <textarea 
              className={styles.instructionArea}
              placeholder="e.g. Always ask the AI to be extremely verbose, logical and format its insights as a technical markdown table."
              value={instruction}
              onChange={(e) => setInstruction(e.target.value)}
            ></textarea>
          </div>
        </div>

        {/* MAIN CHAT AREA */}
        <div className={styles.chatArea}>
          <main className={styles.mainContent} onDragOver={handleDragOver} onDragLeave={handleDragLeave} onDrop={handleDrop}>
            
            {dragging && (
               <div className={styles.uploadAreaDragging} style={{ position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, zIndex: 10, display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'rgba(109, 40, 217, 0.1)' }}>
                  <h3 style={{ color: 'var(--accent-color)' }}><FileText size={32} style={{ verticalAlign: 'middle', marginRight: '8px' }}/> Drop PDF Document Here</h3>
               </div>
            )}

            {messages.length === 0 ? (
                <div className={styles.chatContainer} style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%' }}>
                  <div style={{ textAlign: 'center', opacity: 0.8, padding: '0 1rem' }}>
                      <Sparkles size={48} style={{ marginBottom: '1rem', color: 'var(--accent-color)' }} />
                      <h2 style={{ fontSize: '1.5rem', marginBottom: '0.5rem' }}>Welcome to Stats AI</h2>
                      <p style={{ marginBottom: '1.5rem', color: 'var(--text-secondary)' }}>What area would you like me to specialize in today?</p>
                      
                      <div className={styles.specializationContainer}>
                        <button 
                          className={`${styles.chipButton} ${specialization === 'statistics' ? styles.chipButtonActive : ''}`}
                          onClick={() => setSpecialization('statistics')}
                        >
                          <BookOpen size={14} /> Statistics
                        </button>
                        <button 
                          className={`${styles.chipButton} ${specialization === 'probability' ? styles.chipButtonActive : ''}`}
                          onClick={() => setSpecialization('probability')}
                        >
                          <Cpu size={14} /> Probability
                        </button>
                        <button 
                          className={`${styles.chipButton} ${specialization === 'data_science' ? styles.chipButtonActive : ''}`}
                          onClick={() => setSpecialization('data_science')}
                        >
                          <Sparkles size={14} /> Data Science
                        </button>
                        <button 
                          className={`${styles.chipButton} ${specialization === 'research' ? styles.chipButtonActive : ''}`}
                          onClick={() => setSpecialization('research')}
                        >
                          <FileText size={14} /> Research
                        </button>
                      </div>
                      
                      {!specialization && (
                        <p style={{ marginTop: '2rem', fontSize: '0.8rem', opacity: 0.5 }}>
                          Choose an area above to begin our specialized session.
                        </p>
                      )}
                  </div>
                </div>
            ) : (
              <div className={styles.chatContainer}>
                <div className={styles.messageList}>
                  {messages.map((msg, idx) => (
                    <div key={idx} className={`${styles.messageWrapper} ${msg.role === 'user' ? styles.userMessageWrapper : styles.aiMessageWrapper}`}>
                      {msg.role === 'model' && <div className={styles.aiAvatar}>AI</div>}
                      <div className={`${styles.messageBubble} ${msg.role === 'user' ? styles.userBubble : styles.aiBubble}`}>
                        {msg.role === 'user' ? (
                          <>
                            <p>{msg.originalText || msg.text}</p>
                            {msg.isOptimized && (
                               <div style={{ opacity: 0.3, marginTop: '5px', display: 'flex', alignItems: 'center', fontSize: '0.7rem' }}>
                                <Sparkles size={8} style={{ marginRight: '4px' }}/> Optimized implicitly
                               </div>
                            )}
                          </>
                        ) : (
                          <div className="markdown-body">
                            {renderMessageContent(msg.text)}
                          </div>
                        )}
                      </div>
                    </div>
                  ))}

                  {appStatus !== 'idle' && (
                    <div className={`${styles.messageWrapper} ${styles.aiMessageWrapper}`}>
                        <div className={styles.aiAvatar}>AI</div>
                        <div className={`${styles.messageBubble} ${styles.aiBubble} ${styles.loadingContainer}`}>
                          <div className={styles.loadingBubble} style={{ padding: '0.5rem' }}>
                            <div className={styles.dotFlashing}></div>
                          </div>
                        </div>
                    </div>
                  )}
                  <div ref={messagesEndRef} />
                </div>
              </div>
            )}
          </main>

          <footer className={styles.footer}>
            {file && (
              <div className={styles.filePreview}>
                <FileText size={18} />
                <span>{file.name}</span>
                <button type="button" onClick={() => setFile(null)} className={styles.removeFileBtn} title="Remove File">
                  <Trash2 size={16} />
                </button>
              </div>
            )}
            
            <form onSubmit={handleSubmit} className={styles.inputArea}>
              <label className={styles.attachBtn} title={messages.length > 0 ? "Upload new PDF" : "Upload PDF"}>
                <Paperclip size={20} />
                <input type="file" accept="application/pdf" onChange={handleFileChange} hidden />
              </label>
              <input 
                type="text" 
                placeholder={file ? 'Ask a question or press send...' : 'Type your message...'} 
                value={input}
                onChange={(e) => setInput(e.target.value)}
                disabled={appStatus !== 'idle' || !specialization}
              />
              <button type="submit" className={styles.sendBtn} disabled={appStatus !== 'idle' || !specialization || (!input.trim() && !file)}>
                <Send size={18} />
              </button>
            </form>
          </footer>
        </div>

        {/* RIGHT SIDEBAR (PIPELINE TRACER) */}
        <div className={`${styles.rightSidebar} ${isLogsOpen ? styles.rightSidebarOpen : ''}`}>
          <div className={styles.header} style={{ textAlign: 'left', borderBottom: '1px solid rgba(255,255,255,0.1)', paddingBottom: '1rem' }}>
            <h2 style={{ fontSize: '1rem', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Sparkles size={16} color="var(--accent-color)" /> Live Tracing Logs
            </h2>
          </div>

          <div className={styles.stepList}>
            {workflowSteps.length === 0 ? (
               <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', fontStyle: 'italic', textAlign: 'center', marginTop: '2rem' }}>
                 Awaiting input...
               </p>
            ) : (
              workflowSteps.map(step => (
                <div key={step.id} className={`${styles.stepItem} ${step.status === 'running' ? styles.stepItemActive : ''}`}>
                  {getStepIcon(step.status)}
                  <div className={styles.stepContent}>
                    <div className={styles.stepTitle}>{step.title}</div>
                    <div className={styles.stepDetails}>{step.details}</div>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

      </div>
    </>
  );
}

export default App;
