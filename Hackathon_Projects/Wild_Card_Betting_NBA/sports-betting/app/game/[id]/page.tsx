// app/game/[id]/page.tsx
import { notFound } from "next/navigation";
import { BettingOptions } from "@/components/betting-options";
import { ScoreCard } from "@/components/score-card";
import { fetchGameById } from "@/utils/fetchGames";

export default async function GamePage({ params }: { params: { id: string } }) {
  // Fetch game data dynamically
  const game = await fetchGameById(params.id);

  if (!game) {
    notFound();
  }

  return (
    <div className="min-h-screen bg-background p-4 md:p-8">
      <div className="mx-auto max-w-7xl space-y-8">
        {/* Score Card */}
        <ScoreCard game={game} />

        
        {/* Betting Section */}
          <div className="order-first lg:order-none">
            <BettingOptions game={game} />
        </div> 
      </div>
    </div>
  );
}

