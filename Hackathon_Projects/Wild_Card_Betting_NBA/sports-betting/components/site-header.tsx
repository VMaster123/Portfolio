import Link from "next/link"
import { ThemeToggle } from "@/components/theme-toggle"
import { Home, BarChart2, Info } from "lucide-react"

export function SiteHeader() {
  return (
    <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container flex h-16 items-center">
        <div className="flex flex-1">
          <Link href="/" className="mr-8 flex items-center space-x-2 text-primary hover:text-primary/90">
            <BarChart2 className="h-6 w-6" />
            <span className="font-bold text-lg">Sports Betting</span>
          </Link>
          <nav className="flex items-center space-x-6 text-sm font-medium">
            <Link href="/games" className="flex items-center space-x-2 transition-colors hover:text-primary">
              <Home className="h-4 w-4" />
              <span>Today's Games</span>
            </Link>
            <Link href="/about" className="flex items-center space-x-2 transition-colors hover:text-primary">
              <Info className="h-4 w-4" />
              <span>About</span>
            </Link>
          </nav>
        </div>
        <div className="flex items-center justify-end space-x-4">
          <ThemeToggle />
        </div>
      </div>
    </header>
  )
}

