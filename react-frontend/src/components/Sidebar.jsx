// import { useContext } from "react";
// import { AppContext } from "../context/AppContext";
// import { FiX } from "react-icons/fi";

// export default function Sidebar() {
//   const {
//     sidebarOpen, setSidebarOpen,
//     selectedModel, setSelectedModel,
//     repoUrl, setRepoUrl,
//     issues, selectedIssue, setSelectedIssue
//   } = useContext(AppContext);

//   return (
//     <div className={`fixed top-0 left-0 h-full w-72 bg-white shadow-lg transform
//       ${sidebarOpen ? "translate-x-0" : "-translate-x-full"} 
//       transition-transform duration-300 p-4 z-50`}>

//       <div className="flex justify-between items-center mb-4">
//         <h2 className="text-lg font-semibold">Settings</h2>
//         <button onClick={() => setSidebarOpen(false)} className="text-xl">
//           <FiX />
//         </button>
//       </div>

//       <div className="space-y-4">
//         <div>
//           <label className="font-medium">Model</label>
//           <select
//             className="w-full p-2 border rounded"
//             value={selectedModel}
//             onChange={(e) => setSelectedModel(e.target.value)}
//           >
//             <option value="gemini">Gemini</option>
//             <option value="groq">Groq</option>
//           </select>
//         </div>

//         <div>
//           <label className="font-medium">Repository URL</label>
//           <input
//             value={repoUrl}
//             onChange={(e) => setRepoUrl(e.target.value)}
//             className="w-full p-2 border rounded"
//           />
//         </div>

//         <div>
//           <label className="font-medium">Issue</label>
//           <select
//             className="w-full p-2 border rounded"
//             value={selectedIssue || ""}
//             onChange={(e) => setSelectedIssue(e.target.value)}
//           >
//             <option value="">Select Issue</option>
//             {issues.map(i => (
//               <option key={i.id} value={i.id}>
//                 {i.title}
//               </option>
//             ))}
//           </select>
//         </div>

//       </div>
//     </div>
//   );
// }



import { useContext, useState, useRef, useEffect } from "react";
import { AppContext } from "../context/AppContext";
import { FiX, FiCpu, FiGithub, FiAlertCircle, FiChevronDown, FiCheck } from "react-icons/fi";

// --- Helper Component: Custom Dropdown ---
// This handles long text wrapping and consistent styling
const CustomSelect = ({ label, value, options, onChange, placeholder = "Select option" }) => {
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef(null);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setIsOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  // Find label for selected value
  const selectedLabel = options.find(opt => opt.value === value)?.label || placeholder;

  return (
    <div className="relative" ref={dropdownRef}>
      {/* Dropdown Trigger Button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className={`w-full p-4 flex justify-between items-center bg-white/40 border border-white/20 hover:bg-white/50 backdrop-blur-md rounded-xl transition-all duration-200 group text-left ${isOpen ? 'ring-2 ring-gray-900/10' : ''}`}
      >
        <span className={`block truncate font-medium ${!value ? 'text-gray-500' : 'text-gray-800'}`}>
          {selectedLabel}
        </span>
        <FiChevronDown className={`text-gray-500 transition-transform duration-200 ${isOpen ? 'rotate-180' : ''}`} />
      </button>

      {/* Dropdown Menu */}
      {isOpen && (
        <div className="absolute z-50 w-full mt-2 bg-white/90 backdrop-blur-xl border border-white/20 rounded-xl shadow-xl max-h-64 overflow-y-auto animate-in fade-in zoom-in-95 duration-100">
          <div className="p-1 space-y-1">
            {options.map((option) => (
              <button
                key={option.value}
                onClick={() => {
                  onChange(option.value);
                  setIsOpen(false);
                }}
                className={`w-full p-3 text-left rounded-lg text-sm transition-colors flex items-start gap-2
                  ${value === option.value ? 'bg-blue-50 text-blue-700 font-semibold' : 'hover:bg-gray-100 text-gray-700'}
                `}
              >
                {/* Checkmark icon for active item */}
                <div className="mt-0.5 min-w-[16px]">
                  {value === option.value && <FiCheck size={16} />}
                </div>
                {/* Text with wrapping enabled */}
                <span className="break-words w-full">{option.label}</span>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default function Sidebar() {
  const {
    sidebarOpen, setSidebarOpen,
    selectedModel, setSelectedModel,
    repoUrl, setRepoUrl,
    issues, selectedIssue, setSelectedIssue
  } = useContext(AppContext);

  // Prepare options for the custom dropdowns
  const modelOptions = [
    { value: "gemini", label: "Gemini 1.5 Pro" },
    { value: "groq", label: "Groq (Llama 3)" }
  ];

  const issueOptions = issues.map(i => ({
    value: i.id,
    label: `#${i.id} - ${i.title}`
  }));

  return (
    <>
      {/* Backdrop */}
      {sidebarOpen && (
        <div 
          className="fixed inset-0 bg-black/10 backdrop-blur-[2px] z-40 transition-opacity"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Main Sidebar Container - Width set to 35% */}
      <div className={`fixed top-0 left-0 h-full w-[35%] min-w-[400px] max-w-[800px] bg-white/60 backdrop-blur-2xl border-r border-white/40 shadow-2xl transform 
        ${sidebarOpen ? "translate-x-0" : "-translate-x-full"} 
        transition-transform duration-300 ease-out z-50 flex flex-col`}>

        {/* --- Header --- */}
        <div className="flex justify-between items-start p-8 border-b border-gray-200/40">
          <div>
            <h1 className="text-3xl font-bold text-gray-900 tracking-tight">DissolveAI</h1>
            <p className="text-sm text-gray-500 font-medium mt-1">Project Configuration</p>
          </div>
          <button 
            onClick={() => setSidebarOpen(false)} 
            className="p-2 rounded-full hover:bg-black/5 text-gray-600 transition-colors mt-1"
          >
            <FiX size={24} />
          </button>
        </div>

        {/* --- Scrollable Content --- */}
        <div className="flex-1 overflow-y-auto p-8 space-y-10 scrollbar-thin scrollbar-thumb-gray-300 scrollbar-track-transparent">

          {/* Section 1: Model */}
          <section className="space-y-4">
            <div className="flex items-center gap-2 text-gray-800">
              <div className="p-2 bg-white rounded-lg shadow-sm">
                <FiCpu size={18} />
              </div>
              <h3 className="font-bold text-sm uppercase tracking-wider text-gray-600">Model Selection</h3>
            </div>
            
            <CustomSelect 
              value={selectedModel}
              onChange={setSelectedModel}
              options={modelOptions}
            />
          </section>

          {/* Section 2: Repository */}
          <section className="space-y-4">
            <div className="flex items-center gap-2 text-gray-800">
              <div className="p-2 bg-white rounded-lg shadow-sm">
                <FiGithub size={18} />
              </div>
              <h3 className="font-bold text-sm uppercase tracking-wider text-gray-600">Source Control</h3>
            </div>

            <div>
              <input
                value={repoUrl}
                onChange={(e) => setRepoUrl(e.target.value)}
                placeholder="https://github.com/username/repo"
                className="w-full p-4 bg-white/40 border border-white/20 backdrop-blur-md rounded-xl text-gray-800 placeholder-gray-500 focus:ring-2 focus:ring-gray-900/10 focus:border-transparent outline-none transition-all shadow-sm font-medium"
              />
            </div>
          </section>

          {/* Section 3: Issues */}
          <section className="space-y-4">
            <div className="flex items-center gap-2 text-gray-800">
              <div className="p-2 bg-white rounded-lg shadow-sm">
                <FiAlertCircle size={18} />
              </div>
              <h3 className="font-bold text-sm uppercase tracking-wider text-gray-600">Active Context Issue</h3>
            </div>

            <CustomSelect 
              value={selectedIssue}
              onChange={setSelectedIssue}
              options={issueOptions}
              placeholder="-- Select an Issue --"
            />
            
            <p className="text-xs text-gray-500 px-2 leading-relaxed">
              Selecting an issue will load its thread comments and description into the AI context window.
            </p>
          </section>

        </div>

        {/* --- Footer --- */}
        <div className="p-6 border-t border-gray-200/40 bg-white/20 backdrop-blur-sm">
          <div className="flex justify-center">
             <span className="text-xs font-semibold text-gray-400 uppercase tracking-widest">DissolveAI v1.2</span>
          </div>
        </div>
      </div>
    </>
  );
}