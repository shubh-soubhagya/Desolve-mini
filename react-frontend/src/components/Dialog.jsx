// import { useContext, useState } from "react";
// import { AppContext } from "../context/AppContext";
// import backend from "../api/api";

// export default function Dialog({ onDone }) {
//   const {
//     selectedModel, setSelectedModel,
//     repoUrl, setRepoUrl,
//     issues, setIssues,
//     selectedIssue, setSelectedIssue,
//     setSummary
//   } = useContext(AppContext);

//   const [loading, setLoading] = useState(false);

//   async function fetchRepo() {
//     if (!repoUrl.trim()) return alert("Enter repository URL");

//     setLoading(true);
//     setIssues([]);
//     setSelectedIssue(null);

//     try {
//       const res = await backend.processRepo({ url: repoUrl.trim(), model: selectedModel });

//       const cleanIssues = res.data.issues || [];
//       setIssues(cleanIssues);
//       setSummary(res.data.summary || "");

//     } catch (err) {
//       alert("Error: " + err.message);
//     }

//     setLoading(false);
//   }

//   return (
//     <div className="max-w-xl mx-auto mt-10 bg-white p-6 rounded shadow">
//       <h1 className="text-2xl font-semibold mb-4">Setup</h1>

//       {/* Model */}
//       <div className="mb-4">
//         <label className="font-medium">Model</label>
//         <select
//           className="w-full border p-2 rounded"
//           value={selectedModel}
//           onChange={(e) => setSelectedModel(e.target.value)}
//         >
//           <option value="gemini">Gemini</option>
//           <option value="groq">Groq</option>
//         </select>
//       </div>

//       {/* Repo URL */}
//       <div className="mb-4">
//         <label className="font-medium">GitHub URL</label>
//         <input
//           className="w-full border p-2 rounded"
//           value={repoUrl}
//           onChange={(e) => setRepoUrl(e.target.value)}
//           placeholder="https://github.com/user/repo"
//         />
//       </div>

//       <button
//         onClick={fetchRepo}
//         className="w-full bg-blue-600 text-white p-2 rounded"
//       >
//         {loading ? "Processing..." : "Fetch Issues"}
//       </button>

//       {/* Issues */}
//       {issues.length > 0 && (
//         <div className="mt-6">
//           <label className="font-medium">Select Issue</label>
//           <select
//             className="w-full border p-2 rounded"
//             value={selectedIssue || ""}
//             onChange={(e) => setSelectedIssue(parseInt(e.target.value))}
//           >
//             <option value="">Select Issue</option>
//             {issues.map((i) => (
//               <option key={i.id} value={i.id}>
//                 #{i.id} {i.title}
//               </option>
//             ))}
//           </select>

//           <button
//             onClick={() => selectedIssue ? onDone() : alert("Select issue")}
//             className="w-full bg-green-600 mt-4 text-white p-2 rounded"
//           >
//             Continue
//           </button>
//         </div>
//       )}
//     </div>
//   );
// }


import { useContext, useState, useRef, useEffect } from "react";
import { AppContext } from "../context/AppContext";
import backend from "../api/api";
import { FiGithub, FiCpu, FiLayers, FiArrowRight, FiLoader, FiAlertCircle, FiCheck, FiChevronDown, FiX } from "react-icons/fi";
import Navbar from "./Navbar"; 

// --- 1. UPDATED: Custom Select (Centered Modal Style) ---
const CustomSelect = ({ value, onChange, options, placeholder, label }) => {
  const [isOpen, setIsOpen] = useState(false);

  // Find label for selected value
  const selectedLabel = options.find(opt => opt.value === value)?.label || placeholder;

  return (
    <>
      {/* Trigger Button (The Input Box) */}
      <button
        onClick={() => setIsOpen(true)}
        className="w-full p-3.5 bg-white/50 border border-white/30 backdrop-blur-md rounded-xl text-left flex justify-between items-center transition-all hover:bg-white/60 focus:ring-2 focus:ring-blue-500/20 group"
      >
        <span className={`block truncate font-medium ${!value ? 'text-gray-500' : 'text-gray-800'}`}>
           {selectedLabel}
        </span>
        <FiChevronDown className="text-gray-500" />
      </button>

      {/* THE SELECTION WINDOW (Centered Overlay) */}
      {isOpen && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center px-4">
          
          {/* Backdrop (Click to close) */}
          <div 
            className="absolute inset-0 bg-gray-900/40 backdrop-blur-sm transition-opacity"
            onClick={() => setIsOpen(false)}
          />

          {/* The Window Itself */}
          <div className="relative w-full max-w-lg bg-white rounded-2xl shadow-2xl overflow-hidden animate-in zoom-in-95 duration-200 border border-gray-100">
            
            {/* Window Header */}
            <div className="px-6 py-4 border-b border-gray-100 flex justify-between items-center bg-gray-50">
              <h3 className="font-bold text-gray-700 uppercase tracking-wide text-sm">
                {label || "Select Option"}
              </h3>
              <button 
                onClick={() => setIsOpen(false)}
                className="p-1 hover:bg-gray-200 rounded-full text-gray-500 transition-colors"
              >
                <FiX size={20} />
              </button>
            </div>

            {/* Scrollable List Area */}
            <div className="max-h-[50vh] overflow-y-auto p-2 scrollbar-thin scrollbar-thumb-gray-300 scrollbar-track-transparent">
              {options.length === 0 ? (
                <div className="p-8 text-center text-gray-400">No options available</div>
              ) : (
                <div className="space-y-1">
                  {options.map((option) => (
                    <button
                      key={option.value}
                      onClick={() => {
                        onChange(option.value);
                        setIsOpen(false);
                      }}
                      className={`w-full p-4 text-left rounded-xl text-sm transition-all flex items-start gap-3 border border-transparent
                        ${value === option.value 
                          ? 'bg-blue-50 border-blue-200 text-blue-700 shadow-sm' 
                          : 'hover:bg-gray-50 text-gray-700 hover:border-gray-200'}
                      `}
                    >
                      {/* Checkmark Icon */}
                      <div className={`mt-0.5 min-w-[20px] flex justify-center ${value === option.value ? 'text-blue-600' : 'text-transparent'}`}>
                        <FiCheck size={18} />
                      </div>
                      
                      {/* Text Content */}
                      <div className="flex-1">
                        <span className="block font-semibold text-base mb-0.5">
                          {option.label.split(' - ')[0]} {/* Grab ID part if available */}
                        </span>
                        <span className="block text-gray-500 font-medium leading-relaxed">
                           {/* If the label has a hyphen, show the rest, otherwise show full label */}
                           {option.label.includes(' - ') ? option.label.split(' - ').slice(1).join(' - ') : option.label}
                        </span>
                      </div>
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </>
  );
};

// --- 2. Main Dialog Component (Unchanged Logic, updated props) ---

const PROCESSING_STEPS = [
  "Processing Request...",
  "Cloning Repository...",
  "Fetching Codes & Understanding...",
  "Cleaning Data & Preprocessing...",
  "Analyzing Contextual Relations...",
  "Fetching Issues...",
  "Creating Knowledge Base..."
];

export default function Dialog({ onDone }) {
  const {
    selectedModel, setSelectedModel,
    repoUrl, setRepoUrl,
    issues, setIssues,
    selectedIssue, setSelectedIssue,
    setSummary
  } = useContext(AppContext);

  const [loading, setLoading] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);

  async function fetchRepo() {
    if (!repoUrl.trim()) return alert("Enter repository URL");

    setLoading(true);
    setCurrentStep(0);
    setIssues([]);
    setSelectedIssue(null);

    const stepInterval = setInterval(() => {
      setCurrentStep((prev) => {
        if (prev < PROCESSING_STEPS.length - 1) return prev + 1;
        return prev;
      });
    }, 800);

    try {
      const res = await backend.processRepo({ url: repoUrl.trim(), model: selectedModel });
      const cleanIssues = res.data.issues || [];
      setIssues(cleanIssues);
      setSummary(res.data.summary || "");
    } catch (err) {
      alert("Error: " + err.message);
    } finally {
      clearInterval(stepInterval);
      setLoading(false);
      setCurrentStep(0);
    }
  }

  const inputClasses = "w-full p-3.5 bg-white/50 border border-white/30 backdrop-blur-md rounded-xl text-gray-800 placeholder-gray-500 focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 outline-none transition-all shadow-sm font-medium";
  const labelClasses = "block text-sm font-bold text-gray-700 mb-1.5 uppercase tracking-wide flex items-center gap-2";

  const modelOptions = [
    { value: "gemini", label: "Gemini 1.5 Pro" },
    { value: "groq", label: "Groq (Llama 3)" }
  ];

  const issueOptions = issues.map(i => ({
    value: i.id,
    label: `#${i.id} - ${i.title}`
  }));

  return (
    <div className="min-h-screen w-full fixed inset-0 z-50 flex flex-col font-sans">
      
      <video autoPlay loop muted className="absolute inset-0 w-full h-full object-cover -z-20">
        <source src="/video2.mp4" type="video/mp4" />
      </video>

      <div className="z-50">
        <Navbar />
      </div>

      <div className="flex-1 flex items-center justify-center p-4 overflow-y-auto">
        <div className="relative w-full max-w-xl bg-white/60 backdrop-blur-2xl border border-white/40 p-8 rounded-3xl shadow-2xl animate-in fade-in zoom-in-95 duration-300">
          
          <div className="mb-8 text-center">
            <h1 className="text-3xl font-extrabold text-gray-900 tracking-tight mb-2">Project Setup</h1>
            <p className="text-gray-600 font-medium">Configure your repository and AI model to begin.</p>
          </div>

          <div className="space-y-6">
            
            <div>
              <label className={labelClasses}><FiCpu /> Model Selection</label>
              <CustomSelect 
                label="Select AI Model"
                value={selectedModel}
                onChange={setSelectedModel}
                options={modelOptions}
              />
            </div>

            <div>
              <label className={labelClasses}><FiGithub /> GitHub Repository URL</label>
              <input
                className={inputClasses}
                value={repoUrl}
                onChange={(e) => setRepoUrl(e.target.value)}
                placeholder="https://github.com/username/repo"
              />
            </div>

            <button
              onClick={fetchRepo}
              disabled={loading}
              className={`w-full py-3.5 px-6 rounded-xl font-bold text-white tracking-wide shadow-lg transition-all duration-300 flex items-center justify-center gap-3
                ${loading 
                  ? "bg-gray-800 cursor-progress ring-2 ring-blue-500/50" 
                  : "bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 hover:shadow-xl active:scale-[0.98]"
                }`}
            >
              {loading ? (
                <>
                  <FiLoader className="animate-spin text-xl text-blue-300" /> 
                  <span className="animate-pulse">{PROCESSING_STEPS[currentStep]}</span>
                </>
              ) : (
                <><FiLayers className="text-xl" /> Fetch Issues & Analyze</>
              )}
            </button>

            {issues.length > 0 && (
              <div className="pt-6 border-t border-gray-200/30 animate-in fade-in slide-in-from-bottom-4 duration-300">
                <label className={labelClasses}><FiAlertCircle /> Select Context Issue</label>
                
                {/* Updated to use the new Centered Modal Select */}
                <div className="mb-4">
                  <CustomSelect 
                    label="Select Context Issue"
                    value={selectedIssue}
                    onChange={setSelectedIssue}
                    options={issueOptions}
                    placeholder="-- Select an Issue to Focus On --"
                  />
                </div>

                <button
                  onClick={() => selectedIssue ? onDone() : alert("Please select an issue to continue.")}
                  className="w-full py-3.5 px-6 bg-gradient-to-r from-emerald-500 to-teal-600 hover:from-emerald-600 hover:to-teal-700 text-white rounded-xl font-bold shadow-lg hover:shadow-xl transition-all active:scale-[0.98] flex items-center justify-center gap-2"
                >
                  Continue to Workspace <FiArrowRight className="text-lg" />
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}