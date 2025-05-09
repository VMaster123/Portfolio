import { Button } from "@/components/ui/button"
import Link from "next/link"
import { AnimatedBackground } from "@/components/background/animated-background"

export default function Home() {
  return (
    <div className="relative flex-1">
      <AnimatedBackground />
      <section className="relative w-full py-12 md:py-24 lg:py-32 xl:py-48">
        <div className="container px-4 md:px-6">
          <div className="flex flex-col items-center space-y-4 text-center">
            <div className="space-y-2">
              <h1 className="text-3xl font-bold tracking-tighter text-white sm:text-4xl md:text-5xl lg:text-6xl/none">
                Live Sports Betting
              </h1>
              <p className="mx-auto max-w-[700px] text-white/80 md:text-xl">
                Your premier destination for real-time sports betting. Watch games, track scores, and place bets with
                live updates and competitive odds.
              </p>
            </div>
            <div className="space-x-4">
              <Link href="/games">
                <Button size="lg" className="bg-primary hover:bg-primary/90">
                  View Today's Games
                </Button>
              </Link>
              <Link href="/about">
                <Button variant="outline" size="lg" className="border-white text-white hover:bg-white/10">
                  Learn More
                </Button>
              </Link>
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}

