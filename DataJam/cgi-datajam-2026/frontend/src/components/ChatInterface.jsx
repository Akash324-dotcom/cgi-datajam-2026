import React, { useState, useRef, useEffect } from 'react';

const ChatInterface = ({ messages, onSendMessage, isLoading, locationStatus }) => {
    const [input, setInput] = useState('');
    const messagesEndRef = useRef(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const handleSubmit = (e) => {
        e.preventDefault();

        if (input.trim() && !isLoading) {
            onSendMessage(input.trim());
            setInput('');
        }
    };

    const getUrgencyColor = (level) => {
        switch (level) {
            case 1: return 'bg-red-600 text-white'; // ER Immediate
            case 2: return 'bg-yellow-500 text-white'; // ER
            case 3: return 'bg-orange-500 text-white'; // Urgent Clinic
            case 4: return 'bg-green-600 text-white'; // Clinic
            case 5: return 'bg-blue-400 text-slate-800'; // Non-Urgent (Pharmacy)
            default: return 'bg-slate-200 text-slate-800';
        }
    };

    const LocationIndicator = () => {
        if (locationStatus === 'granted') {
            return (
                <span className="text-[10px] text-green-300 flex items-center mt-1">
                    <i className="fas fa-location-arrow mr-1"></i> Location Active
                </span>
            );
        }
        else if (locationStatus === 'pending') {
            return (
                <span className="text-[10px] text-amber-300 flex items-center mt-1">
                    <i className="fas fa-location-dot mr-1"></i> Requesting Location
                </span>
            );
        }
        else {
            return (
                <span className="text-[10px] text-amber-300 flex items-center mt-1" title="Using default Nova Scotia center">
                    <i className="fas fa-map-marker-alt mr-1"></i> Default Location - Nova Scotia Province
                </span>
            );
        }
    };

    return (
        <div className="flex flex-col h-full bg-white border-r border-slate-200 shadow-xl z-20">
            <div className="p-5 border-b border-slate-200 bg-gradient-to-r from-blue-900 to-blue-800 text-white flex items-center justify-between">
                <div>
                    <h2 className="text-xl font-bold flex items-center tracking-tight">
                        <i className="fas fa-heartbeat mr-3 text-red-400 text-2xl"></i>
                        <span>NS Health Navigator</span>
                    </h2>
                    <p className="text-xs text-blue-200 mt-1 font-medium">AI Triage & Predictive Routing Engine</p>
                </div>
                <div className="bg-blue-700/50 p-2 rounded-lg text-xs font-semibold backdrop-blur-sm border border-blue-600/50">
                    <i className="fas fa-shield-halved mr-1"></i> CTAS Aligned
                </div>
            </div>

            <div className="flex-1 overflow-y-auto p-4 space-y-6 bg-slate-50">
                {messages.length === 0 && (
                    <div className="h-full flex flex-col items-center justify-center text-center px-6 opacity-70">
                        <div className="w-16 h-16 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center mb-4 text-3xl">
                            <i className="fas fa-stethoscope"></i>
                        </div>
                        <h3 className="text-lg font-bold text-slate-700 mb-2">Welcome to Nova Health</h3>
                        <p className="text-sm text-slate-500">
                            Describe your symptoms below. Our system will assess your condition based on Canadian Triage guidelines and route you to the fastest appropriate care.
                        </p>
                    </div>
                )}

                {messages.map((msg) => (
                    <div key={msg.id} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                        <div className={`max-w-[85%] rounded-2xl p-4 shadow-sm ${msg.role === 'user'
                                ? 'bg-blue-600 text-white rounded-tr-none'
                                : 'bg-white text-slate-800 rounded-tl-none border border-slate-200'
                            }`}>
                            {msg.role === 'model' && msg.triageData && msg.triageData.level > 0 && msg.triageData.level <= 5 && (
                                <div className={`text-xs font-bold px-3 py-1.5 rounded-full inline-flex items-center mb-3 shadow-sm ${getUrgencyColor(msg.triageData.level)}`}>
                                    <i className="fas fa-wave-square mr-2"></i>
                                    CTAS Level {msg.triageData.level}: {msg.triageData.urgency}
                                </div>
                            )}

                            <div className="prose prose-sm max-w-none whitespace-pre-wrap leading-relaxed">
                                {msg.role === 'user' ? msg.content : (msg.triageData?.reply || msg.content)}
                            </div>

                            <div className={`text-[10px] mt-3 font-medium ${msg.role === 'user' ? 'text-blue-200 text-right' : 'text-slate-400 text-left'}`}>
                                {msg.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                            </div>
                        </div>
                    </div>
                ))}

                {isLoading && (
                    <div className="flex justify-start">
                        <div className="bg-white rounded-2xl rounded-tl-none p-5 border border-slate-200 shadow-sm flex items-center space-x-3">
                            <div className="w-2.5 h-2.5 bg-blue-500 rounded-full animate-bounce"></div>
                            <div className="w-2.5 h-2.5 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '0.15s' }}></div>
                            <div className="w-2.5 h-2.5 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '0.3s' }}></div>
                            <span className="text-xs font-medium text-slate-400 ml-2">Triaging symptoms...</span>
                        </div>
                    </div>
                )}
                <div ref={messagesEndRef} />
            </div>

            <div className="p-4 bg-white border-t border-slate-200 shadow-[0_-4px_6px_-1px_rgba(0,0,0,0.05)]">
                <form onSubmit={handleSubmit} className="relative flex items-end">
                    <textarea
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyDown={(e) => {
                            if (e.key === 'Enter' && !e.shiftKey) {
                                e.preventDefault();
                                handleSubmit(e);
                            }
                        }}
                        placeholder="E.g., I have had a severe headache for 2 days..."
                        className="w-full pl-4 pr-14 py-3 bg-slate-50 border border-slate-300 rounded-2xl focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none h-14 max-h-32 transition-all shadow-inner"
                        disabled={isLoading}
                        rows={1}
                    />
                    <button
                        type="submit"
                        disabled={!input.trim() || isLoading}
                        className="absolute right-2 bottom-2 p-2 w-10 h-10 flex items-center justify-center bg-blue-600 text-white rounded-xl hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors shadow-md"
                    >
                        <i className="fas fa-paper-plane"></i>
                    </button>
                </form>
                <p className="text-[10px] text-center mt-3 text-slate-500 font-medium">
                    <i className="fas fa-triangle-exclamation text-yellow-500 mr-1"></i>
                    <span>Not a substitute for medical advice. In an emergency, call <strong className="text-red-500">911</strong>.</span>
                </p>
            </div>
        </div>
    );
};

export default ChatInterface;