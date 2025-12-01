import { useState } from "react";
import Navbar from "./components/Navbar";
import Sidebar from "./components/Sidebar";
import DialogPage from "./pages/DialogPage";
import SummaryPage from "./pages/SummaryPage";
import ChatPage from "./pages/ChatPage";

export default function App() {
  const [step, setStep] = useState(1);

  return (
    <>
      <Navbar />
      <Sidebar />

      {step === 1 && <DialogPage onDone={() => setStep(2)} />}
      {step === 2 && <SummaryPage onStartChat={() => setStep(3)} />}
      {step === 3 && <ChatPage />}
    </>
  );
}
