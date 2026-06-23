"use client";

import { useState, useEffect } from "react";

interface StrikethroughTyperProps {
  /**
   * e.g. ["1 Jam", "30 Menit", "5 Menit"]
   * Last word pauses without strikethrough, then loops back.
   */
  words: string[];
  typingSpeed?: number;
  strikeDelay?: number;
  /** Pause on last word (the "winning" word) before looping */
  finalPause?: number;
  nextDelay?: number;
  className?: string;
  finalClassName?: string;
}

type Phase =
  | "typing"
  | "pause-before-strike"
  | "striking"
  | "pause-after-strike"
  | "show-final"
  | "fade-out";

export default function StrikethroughTyper({
  words,
  typingSpeed = 90,
  strikeDelay = 600,
  finalPause = 2500,
  nextDelay = 400,
  className = "",
  finalClassName = "",
}: StrikethroughTyperProps) {
  const [wordIndex, setWordIndex] = useState(0);
  const [charIndex, setCharIndex] = useState(0);
  const [phase, setPhase] = useState<Phase>("typing");
  const [strikeProgress, setStrikeProgress] = useState(0);
  const [opacity, setOpacity] = useState(1);

  const isLastWord = wordIndex === words.length - 1;
  const currentWord = words[wordIndex] ?? "";
  const displayedText = currentWord.slice(0, charIndex);

  // Reset opacity when entering typing phase
  useEffect(() => {
    if (phase === "typing") setOpacity(1);
  }, [phase]);

  // ── Typing ──
  useEffect(() => {
    if (phase !== "typing") return;
    if (charIndex < currentWord.length) {
      const t = setTimeout(() => setCharIndex((c) => c + 1), typingSpeed);
      return () => clearTimeout(t);
    }
    // Finished typing this word
    if (isLastWord) {
      setPhase("show-final");
    } else {
      setPhase("pause-before-strike");
    }
  }, [phase, charIndex, currentWord, typingSpeed, isLastWord]);

  // ── Pause before striking ──
  useEffect(() => {
    if (phase !== "pause-before-strike") return;
    const t = setTimeout(() => {
      setPhase("striking");
      setStrikeProgress(0);
    }, strikeDelay);
    return () => clearTimeout(t);
  }, [phase, strikeDelay]);

  // ── Striking animation ──
  useEffect(() => {
    if (phase !== "striking") return;
    if (strikeProgress < 100) {
      const t = setTimeout(() => setStrikeProgress((p) => Math.min(p + 8, 100)), 25);
      return () => clearTimeout(t);
    }
    setPhase("pause-after-strike");
  }, [phase, strikeProgress]);

  // ── Pause after strike → next word ──
  useEffect(() => {
    if (phase !== "pause-after-strike") return;
    const t = setTimeout(() => {
      setWordIndex((i) => i + 1);
      setCharIndex(0);
      setStrikeProgress(0);
      setPhase("typing");
    }, nextDelay);
    return () => clearTimeout(t);
  }, [phase, nextDelay]);

  // ── Show final word (no strike) → then fade out & loop ──
  useEffect(() => {
    if (phase !== "show-final") return;
    const t = setTimeout(() => setPhase("fade-out"), finalPause);
    return () => clearTimeout(t);
  }, [phase, finalPause]);

  // ── Fade out → reset to first word ──
  useEffect(() => {
    if (phase !== "fade-out") return;
    setOpacity(0);
    const t = setTimeout(() => {
      setWordIndex(0);
      setCharIndex(0);
      setStrikeProgress(0);
      setPhase("typing");
    }, 500); // matches CSS transition duration
    return () => clearTimeout(t);
  }, [phase]);

  const isBeingStruck = phase === "striking" || phase === "pause-after-strike";

  return (
    <span className={className}>
      <span
        className={`relative inline-block transition-opacity duration-500 ${
          phase === "show-final" ? finalClassName : ""
        }`}
        style={{ opacity }}
      >
        {/* Text */}
        <span className={isBeingStruck ? "opacity-40" : ""}>
          {displayedText}
        </span>

        {/* Animated strike line */}
        {isBeingStruck && (
          <span
            className="absolute left-0 top-1/2 h-[3px] bg-current rounded-full"
            style={{
              width: `${strikeProgress}%`,
              transform: "translateY(-50%)",
              opacity: 0.6,
            }}
          />
        )}
      </span>

      {/* Blinking cursor */}
      <span
        className="inline-block w-[3px] ml-0.5 rounded-full animate-blink"
        style={{
          height: "0.85em",
          verticalAlign: "text-bottom",
          backgroundColor: "currentColor",
          opacity,
          transition: "opacity 500ms",
        }}
      />
    </span>
  );
}
