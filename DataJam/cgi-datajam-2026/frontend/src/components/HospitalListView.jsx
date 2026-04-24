import React from 'react';
import HospitalCard from './HospitalCard';

const HospitalListView = ({ hospitals }) => {
    return (
        <div className="flex flex-col h-full overflow-y-auto">
            <div className="flex-shrink-0">
                <h2 className="text-xl font-bold text-slate-800 mb-4">Recommended Facilities</h2>
                <div className="grid gap-4 md:grid-cols-1 lg:grid-cols-2 mb-4">
                    {hospitals.map((hospital, index) => (
                        <HospitalCard key={index} {...hospital} />
                    ))}
                </div>
            </div>

            <div className="flex-grow"></div>

            {/* Informational Cards - Always at bottom */}
            <div className="flex-shrink-0 mt-4">
                <div className="grid gap-4 md:grid-cols-1 lg:grid-cols-3">
                    {/* CTAS Levels Card */}
                    <div className="bg-white rounded-lg shadow-md p-4 border border-slate-200">
                        <h3 className="font-bold text-lg text-slate-800 mb-3 flex items-center">
                            <i className="fas fa-exclamation-triangle text-red-500 mr-2"></i>
                            CTAS Levels
                        </h3>
                        <div className="space-y-2 text-sm">
                            <div className="flex items-center">
                                <span className="w-6 h-6 bg-red-600 text-white text-xs font-bold rounded-full flex items-center justify-center mr-2">1</span>
                                <span className="text-slate-700">Resuscitation - Life threatening</span>
                            </div>
                            <div className="flex items-center">
                                <span className="w-6 h-6 bg-yellow-500 text-white text-xs font-bold rounded-full flex items-center justify-center mr-2">2</span>
                                <span className="text-slate-700">Emergent - Serious condition</span>
                            </div>
                            <div className="flex items-center">
                                <span className="w-6 h-6 bg-orange-500 text-white text-xs font-bold rounded-full flex items-center justify-center mr-2">3</span>
                                <span className="text-slate-700">Urgent - Needs timely care</span>
                            </div>
                            <div className="flex items-center">
                                <span className="w-6 h-6 bg-green-600 text-white text-xs font-bold rounded-full flex items-center justify-center mr-2">4</span>
                                <span className="text-slate-700">Less Urgent - Can wait</span>
                            </div>
                            <div className="flex items-center">
                                <span className="w-6 h-6 bg-blue-400 text-slate-800 text-xs font-bold rounded-full flex items-center justify-center mr-2">5</span>
                                <span className="text-slate-700">Non-Urgent - Self-care or pharmacy</span>
                            </div>
                        </div>
                    </div>

                    {/* Family Doctors Card */}
                    <div className="bg-white rounded-lg shadow-md p-4 border border-slate-200">
                        <h3 className="font-bold text-lg text-slate-800 mb-3 flex items-center">
                            <i className="fas fa-user-md text-blue-500 mr-2"></i>
                            Family Doctors
                        </h3>
                        <p className="text-sm text-slate-600 mb-2">
                            Family doctors (GPs) provide primary care for routine health needs, preventive care, and manage chronic conditions.
                        </p>
                        <p className="text-sm text-slate-600 mb-3">
                            <strong>How to get one:</strong> Register with a family practice through your provincial health plan. Many communities have walk-in clinics while you search.
                        </p>
                        <a href="https://www.nshealth.ca/family-doctors" target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:text-blue-800 text-sm font-medium">
                            Find a Family Doctor →
                        </a>
                    </div>

                    {/* Insurance Resources Card */}
                    <div className="bg-white rounded-lg shadow-md p-4 border border-slate-200">
                        <h3 className="font-bold text-lg text-slate-800 mb-3 flex items-center">
                            <i className="fas fa-shield-alt text-green-500 mr-2"></i>
                            Health Insurance
                        </h3>
                        <p className="text-sm text-slate-600 mb-2">
                            Nova Scotia has universal healthcare through Medicare. International students may need supplemental coverage.
                        </p>
                        <div className="space-y-1 text-sm">
                            <a href="https://www.nshealth.ca/medicare" target="_blank" rel="noopener noreferrer" className="block text-blue-600 hover:text-blue-800">Medicare Information</a>
                            <a href="https://www.canada.ca/en/services/health.html" target="_blank" rel="noopener noreferrer" className="block text-blue-600 hover:text-blue-800">Federal Health Resources</a>
                            <a href="https://www.nshealth.ca/international" target="_blank" rel="noopener noreferrer" className="block text-blue-600 hover:text-blue-800">International Visitor Insurance</a>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default HospitalListView;