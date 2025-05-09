import { fetchLiveGames } from "@/utils/fetchGames"
import { GameCard } from "@/components/game-card"

export default async function GamesPage() {
  const games = await fetchLiveGames()

  return (
    <div className="min-h-screen bg-background p-4 md:p-8">
      <div className="mx-auto max-w-4xl">
        <h1 className="mb-8 text-3xl font-bold tracking-tight">Today's Games</h1>
        {games.length > 0 ? (
          <div className="space-y-4">
            {games.map((game: any) => (
              <GameCard key={game.id} game={game} />
            ))}
          </div>
        ) : (
          <p>No live games available.</p>
        )}
      </div>
    </div>
  )
}

