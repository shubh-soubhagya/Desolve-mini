import { createContext, useState } from "react";

export const AppContext = createContext();

export default function AppProvider({ children }) {
  const [selectedModel, setSelectedModel] = useState("gemini");
  const [repoUrl, setRepoUrl] = useState("");
  const [issues, setIssues] = useState([]);
  const [selectedIssue, setSelectedIssue] = useState(null);
  const [summary, setSummary] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [chatMessages, setChatMessages] = useState([]);

  return (
    <AppContext.Provider value={{
      selectedModel, setSelectedModel,
      repoUrl, setRepoUrl,
      issues, setIssues,
      selectedIssue, setSelectedIssue,
      summary, setSummary,
      sidebarOpen, setSidebarOpen,
      chatMessages, setChatMessages
    }}>
      {children}
    </AppContext.Provider>
  );
}
