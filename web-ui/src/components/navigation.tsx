"use client"

import Link from "next/link"
import Image from "next/image"

// Removed navigation array as all items were non-functional links

export function Navigation() {

  return (
    <header className="sticky top-0 z-50 w-full border-b border-border/40 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container mx-auto px-4 sm:px-6 flex h-14 max-w-screen-2xl items-center">
        <div className="mr-4 flex">
          <div className="mr-2 sm:mr-6 flex items-center space-x-2">
            <Link 
              href="https://www.kebayorantechnologies.com" 
              target="_blank" 
              rel="noopener noreferrer"
              className="flex items-center"
            >
              <Image 
                src="/logo_keytech.png" 
                alt="Keytech Logo" 
                width={32} 
                height={32} 
                className="h-8 w-8 hover:opacity-80 transition-opacity" 
              />
            </Link>
            <Link href="/" className="flex items-center">
              <span className="font-bold text-sm sm:text-base">
                Onegin: A Neural Architecture Explorer
              </span>
            </Link>
          </div>
          {/* Removed navigation items as they linked to non-existent pages */}
        </div>
        
        <div className="flex flex-1 items-center justify-between space-x-2 md:justify-end">
          <div className="w-full flex-1 md:w-auto md:flex-none">
            {/* Search or additional controls can go here */}
          </div>
          {/* Removed Settings icon as it links to non-existent page */}
        </div>
      </div>
    </header>
  )
}