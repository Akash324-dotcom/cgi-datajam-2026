import React, { useState, useEffect } from 'react';
import ChatInterface from './components/ChatInterface';
// import MapView from './components/MapView';
import HospitalListView from './components/HospitalListView';
import { sendTriageRequest } from './services/service';

const App = () => {
    // Initial Mock hospital data
    const INITIAL_HOSPITALS = [
        {
            facility: "Halifax Infirmary (QEII)",
            address: "1796 Summer Street",
            town: "Halifax",
            county: "Halifax",
            type: "regional",
            the_geom: "POINT(-63.5714 44.6488)"
        },
        {
            facility: "Dartmouth General Hospital",
            address: "325 Pleasant Street",
            town: "Dartmouth",
            county: "Halifax",
            type: "regional",
            the_geom: "POINT(-63.5156 44.6688)"
        },
        {
            facility: "IWK Health Centre",
            address: "5850/5980 University Avenue",
            town: "Halifax",
            county: "Halifax",
            type: "regional",
            the_geom: "POINT(-63.5920 44.6361)"
        },
        {
            facility: "Cobequid Community Health Centre",
            address: "40 Freer Lane",
            town: "Lower Sackville",
            county: "Halifax",
            type: "community",
            the_geom: "POINT(-63.6789 44.7765)"
        }
    ];

    const [messages, setMessages] = useState([]);
    const [isLoading, setIsLoading] = useState(false);
    const [latestTriage, setLatestTriageData] = useState(null);
    const [userLocation, setUserLocation] = useState(null);
    const [locationStatus, setLocationStatus] = useState('pending'); // 'pending', 'granted', 'denied', 'error', 'unsupported'
    const [hospitals, setHospitals] = useState(INITIAL_HOSPITALS);
    const [viewMode, setViewMode] = useState('list'); // 'list' or 'map'

    useEffect(() => {
        if ("geolocation" in navigator) {
            navigator.geolocation.getCurrentPosition (
                (position) => {
                    setUserLocation({
                        lat: position.coords.latitude,
                        lng: position.coords.longitude
                    });
                    setLocationStatus('granted');
                },
                (error) => {
                    console.error('Error getting location:', error);
                    setLocationStatus(error.code === 1 ? 'denied' : 'error'); // 1 = permission denied
                },
                { enableHighAccuracy: true, timeout: 10000, maximumAge: 0 }
            );
        }
        else {
            setLocationStatus('unsupported');
        }
    }, []);

    const handleSendMessage = async (text) => {
        const userMessage = {
            id: Date.now(),
            role: 'user',
            content: text,
            timestamp: new Date()
        };

        setMessages((prev) => [...prev, userMessage]);
        setIsLoading(true);

        // send request to backend
        try {
            const triageData = await sendTriageRequest(text);
            setLatestTriageData(triageData);

          
            const modelMessage = {
                id: Date.now() + 1,
                role: 'model',
                content: triageData.reply || 'No debug reply returned.',
                triageData,
                timestamp: new Date()
            };

            setMessages((prev) => [...prev, modelMessage]);
        } catch (error) {
            const errorMessage = {
                id: Date.now() + 1,
                role: 'model',
                content: 'I could not reach the triage service right now. Please try again in a moment.',
                timestamp: new Date()
            };

            setMessages((prev) => [...prev, errorMessage]);
            console.error('Triage request failed:', error);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="flex flex-col md:flex-row h-full w-full font-sans bg-slate-50 text-slate-900 overflow-hidden">
            {/* Left Panel: AI Triage Chat */}
            <div className="w-full md:w-[400px] lg:w-[450px] h-1/2 md:h-full flex-shrink-0 z-10 shadow-2xl transition-all duration-300">
                <ChatInterface
                    messages={messages}
                    onSendMessage={handleSendMessage}
                    isLoading={isLoading}
                    locationStatus={locationStatus}
                />
            </div>

            {/* Right Panel: Hospital Recommendations & Map */}
            <div className="w-full flex-1 h-1/2 md:h-full relative z-0 border-t md:border-t-0 md:border-l border-slate-300 shadow-inner p-4 flex flex-col" style={{ backgroundImage: 'linear-gradient(90deg, rgba(148, 163, 184, 0.1) 1px, transparent 1px), linear-gradient(0deg, rgba(148, 163, 184, 0.1) 1px, transparent 1px)', backgroundSize: '20px 20px' }}>
                
                {/* Navigation Toolbar */}
                <div className="flex-shrink-0 mb-4 bg-slate-50/80 backdrop-blur-sm p-1 rounded-lg border border-slate-200 inline-flex self-start sticky top-0 z-20">
                    <button 
                        onClick={() => setViewMode('list')}
                        className={`px-4 py-2 rounded-md text-sm font-medium transition-all duration-200 ${
                            viewMode === 'list' 
                                ? 'bg-blue-600 text-white shadow-sm' 
                                : 'text-slate-600 hover:bg-slate-200 hover:text-slate-900'
                        }`}
                    >
                        <i className="fas fa-list-ul mr-2"></i> List View
                    </button>
                    {/* <button 
                        onClick={() => setViewMode('map')}
                        className={`px-4 py-2 rounded-md text-sm font-medium transition-all duration-200 ${
                            viewMode === 'map' 
                                ? 'bg-blue-600 text-white shadow-sm' 
                                : 'text-slate-600 hover:bg-slate-200 hover:text-slate-900'
                        }`}
                    >
                        <i className="fas fa-map-marked-alt mr-2"></i> Map View
                    </button> */}
                </div>

                {/* Content Area */}
                <div className="flex-grow overflow-hidden relative">
                    {viewMode === 'list' ? (
                        <HospitalListView hospitals={hospitals} />
                    ) : (
                        <MapView triageData={latestTriage} userLocation={userLocation} />
                    )}
                </div>
            </div>

        </div>
    );
};

export default App;
