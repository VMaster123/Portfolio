"use client"

import Link from "next/link"
import { Card, CardHeader } from "@/components/ui/card"

interface Game {
  id: string
  homeTeam: string
  awayTeam: string
  homeScore: number
  awayScore: number
  status: string
  quarter: string
  timeLeft: string
}

interface GameCardProps {
  game: Game
}

export function GameCard({ game }: GameCardProps) {
  return (
    <Link href={`/game/${game.id}`}>
      <Card className="transition-all hover:scale-[1.02] hover:shadow-lg border-primary/20 hover:border-primary">
        <CardHeader>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div className="space-y-1">
                <p className="text-sm text-primary">{game.status}</p>
                <p className="text-xs text-muted-foreground">
                  {game.quarter} • {game.timeLeft}
                </p>
              </div>
            </div>
            <div className="grid grid-cols-[1fr,auto,1fr] items-center gap-4">
              <div>
                <p className="font-semibold">{game.homeTeam}</p>
                <p className="text-2xl font-bold">{game.homeScore}</p>
              </div>
              <div className="text-sm text-primary font-medium">VS</div>
              <div className="text-right">
                <p className="font-semibold">{game.awayTeam}</p>
                <p className="text-2xl font-bold">{game.awayScore}</p>
              </div>
            </div>
          </div>
        </CardHeader>
      </Card>
    </Link>
  )
}

