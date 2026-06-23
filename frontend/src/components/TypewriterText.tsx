"use client";

import { useState, useEffect, useCallback } from "react";

interface TypewriterTextProps {
  words: string[];
  typingSpeed?: number;
  deletingSpeed?: number;
  pauseDuration?: number;
  className?: string;
}

export default function TypewriterText({
  words,
  typingSpeed = 120,
  deletingSpeed = 60,
  pauseDuration = 2000,
  className = "",
}: TypewriterTextProps) {
  const [wordIndex, setWordIndex] = useState(0);
  const [text, setText] = useState("");
  const [isDeleting, setIsDeleting] = useState(false);
  const [isPaused, setIsPaused] = useState(false);

  const currentWord = words[wordIndex];

  const tick = useCallback(() => {
    if (isPaused) return;

    if (!isDeleting) {
      // Typing
      const next = currentWord.slice(0, text.length + 1);
      setText(next);

      if (next === currentWord) {
        setIsPaused(true);
        setTimeout(() => {
          setIsPaused(false);
          setIsDeleting(true);
        }, pauseDuration);
      }
    } else {
      // Deleting
      const next = currentWord.slice(0, text.length - 1);
      setText(next);

      if (next === "") {
        setIsDeleting(false);
        setWordIndex((prev) => (prev + 1) % words.length);
      }
    }
  }, [text, isDeleting, isPaused, currentWord, words.length, pauseDuration]);

  useEffect(() => {
    const speed = isDeleting ? deletingSpeed : typingSpeed;
    const timer = setTimeout(tick, speed);
    return () => clearTimeout(timer);
  }, [tick, isDeleting, deletingSpeed, typingSpeed]);

  return (
    <span className={className}>
      {text}
      <span
        className="inline-block w-[3px] ml-0.5 rounded-full animate-blink"
        style={{
          height: "0.85em",
          verticalAlign: "text-bottom",
          backgroundColor: "currentColor",
        }}
      />
    </span>
  );
}
