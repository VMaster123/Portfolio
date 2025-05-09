interface Game {
  homeTeam: string
  awayTeam: string
  homeScore: number
  awayScore: number
  status: string
  quarter: string
  timeLeft: string
}

interface ScoreCardProps {
  game: Game
}

export function ScoreCard({ game }: ScoreCardProps) {
  return (
    <div className="rounded-lg border border-primary/20 bg-card p-6 text-card-foreground shadow-lg">
      <div className="mb-4">
        <h1 className="text-3xl font-bold tracking-tight">
          <span className="text-primary">{game.homeTeam}</span>
          <span className="mx-2 text-muted-foreground">vs</span>
          <span className="text-primary">{game.awayTeam}</span>
        </h1>
      </div>
      <div className="grid grid-cols-[1fr,auto,1fr] items-center gap-8">
        <div>
          <p className="text-xl font-semibold text-primary">{game.homeTeam}</p>
          <p className="text-4xl font-bold">{game.homeScore}</p>
        </div>
        <div className="text-center">
          <p className="text-lg font-medium text-primary">VS</p>
          <p className="text-sm text-muted-foreground">
            {game.quarter} • {game.timeLeft}
          </p>
          <p className="text-sm font-medium text-primary">{game.status}</p>
        </div>
        <div className="text-right">
          <p className="text-xl font-semibold text-primary">{game.awayTeam}</p>
          <p className="text-4xl font-bold">{game.awayScore}</p>
        </div>
      </div>
    </div>
  )
}

