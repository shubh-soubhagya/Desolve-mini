import { useState } from "react";
import Dialog from "../components/Dialog";

export default function Home({ onSetupDone }) {
  return <Dialog onDone={onSetupDone} />;
}
