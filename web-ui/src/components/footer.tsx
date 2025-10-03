"use client"

import Link from "next/link"
import Image from "next/image"

const basePath = process.env.NEXT_PUBLIC_BASE_PATH || '';

export function Footer() {
  return (
    <footer className="sticky bottom-0 w-full bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 border-t border-border/40 py-2">
      <div className="container mx-auto px-4 sm:px-6">
        <div className="flex items-center justify-center gap-x-3 text-xs text-muted-foreground">
          <span>© 2025, <Link
            href="https://www.kebayorantechnologies.com"
            target="_blank"
            rel="noopener noreferrer"
            className="text-blue-500 hover:text-blue-700 transition-colors underline"
          >
            Kebayoran Technologies
          </Link> and <Link
            href="https://www.mattmcdonnell.net"
            target="_blank"
            rel="noopener noreferrer"
            className="text-blue-500 hover:text-blue-700 transition-colors underline"
          >
            Matthew McDonnell
          </Link></span>
          <Link
            href="https://github.com/TheBuleGanteng/computer-vision-classification"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center hover:opacity-80 transition-opacity"
          >
            <Image
              src={`${basePath}/github_dark.png`}
              alt="GitHub Repository"
              width={16}
              height={16}
              className="h-4 w-4"
            />
          </Link>
        </div>
      </div>
    </footer>
  )
}