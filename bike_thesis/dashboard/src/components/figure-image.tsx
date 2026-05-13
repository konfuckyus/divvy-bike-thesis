"use client";

import Image from "next/image";
import { useState } from "react";
import { ImageOff } from "lucide-react";

import { cn } from "@/lib/utils";

interface FigureImageProps {
  src: string;
  alt: string;
  className?: string;
  fallbackMessage?: string;
}

export function FigureImage({
  src,
  alt,
  className,
  fallbackMessage,
}: FigureImageProps) {
  const [errored, setErrored] = useState(false);

  if (errored) {
    return (
      <div
        className={cn(
          "flex aspect-[16/10] w-full flex-col items-center justify-center gap-2 rounded-lg border border-dashed border-border bg-muted/40 p-6 text-center",
          className,
        )}
      >
        <ImageOff className="h-8 w-8 text-muted-foreground" />
        <p className="text-sm font-medium text-foreground">
          Figure not available
        </p>
        {fallbackMessage ? (
          <p className="max-w-sm text-xs text-muted-foreground">
            {fallbackMessage}
          </p>
        ) : (
          <p className="max-w-sm text-xs text-muted-foreground">
            The expected image could not be loaded. The file may be missing from
            <code className="mx-1 rounded bg-secondary px-1 py-0.5 text-xs">
              /public/figures
            </code>
            .
          </p>
        )}
        <p className="text-[11px] text-muted-foreground/70">{src}</p>
      </div>
    );
  }

  return (
    <div
      className={cn(
        "relative w-full overflow-hidden rounded-lg border border-border bg-white",
        className,
      )}
    >
      <Image
        src={src}
        alt={alt}
        width={1600}
        height={1000}
        sizes="(min-width: 1280px) 1100px, (min-width: 768px) 80vw, 100vw"
        className="h-auto w-full"
        onError={() => setErrored(true)}
        priority={false}
      />
    </div>
  );
}
