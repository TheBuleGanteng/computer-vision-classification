"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { cn } from "@/lib/utils"
import { 
  BarChart3, 
  Cpu, 
  Home, 
  Settings,
  Activity,
  Layers
} from "lucide-react"

const navigation = [
  {
    name: "Dashboard",
    href: "/",
    icon: Home,
    description: "Overview of all optimization sessions"
  },
  {
    name: "Architecture Explorer",
    href: "/architecture",
    icon: Layers,
    description: "Interactive 3D neural network visualization"
  },
  {
    name: "Trial Comparison",
    href: "/trials",
    icon: BarChart3,
    description: "Compare trials and analyze performance"
  },
  {
    name: "Performance Analytics",
    href: "/analytics",
    icon: Activity,
    description: "Deep dive into optimization metrics"
  },
  {
    name: "System Health",
    href: "/system",
    icon: Cpu,
    description: "Monitor system resources and status"
  }
]

export function Navigation() {
  const pathname = usePathname()

  return (
    <header className="sticky top-0 z-50 w-full border-b border-border/40 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container flex h-14 max-w-screen-2xl items-center">
        <div className="mr-4 hidden md:flex">
          <Link href="/" className="mr-6 flex items-center space-x-2">
            <Layers className="h-6 w-6" />
            <span className="hidden font-bold sm:inline-block">
              Neural Architecture Explorer
            </span>
          </Link>
          <nav className="flex items-center gap-6 text-sm">
            {navigation.map((item) => {
              const Icon = item.icon
              const isActive = pathname === item.href
              
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={cn(
                    "flex items-center gap-2 transition-colors hover:text-foreground/80",
                    isActive 
                      ? "text-foreground font-medium" 
                      : "text-foreground/60"
                  )}
                >
                  <Icon className="h-4 w-4" />
                  <span className="hidden lg:inline-block">{item.name}</span>
                </Link>
              )
            })}
          </nav>
        </div>
        
        <div className="flex flex-1 items-center justify-between space-x-2 md:justify-end">
          <div className="w-full flex-1 md:w-auto md:flex-none">
            {/* Search or additional controls can go here */}
          </div>
          <nav className="flex items-center">
            <Link
              href="/settings"
              className={cn(
                "flex items-center gap-2 px-3 py-2 text-sm transition-colors hover:text-foreground/80",
                pathname === "/settings" 
                  ? "text-foreground font-medium" 
                  : "text-foreground/60"
              )}
            >
              <Settings className="h-4 w-4" />
              <span className="sr-only">Settings</span>
            </Link>
          </nav>
        </div>
      </div>
    </header>
  )
}