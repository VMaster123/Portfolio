// components/betting-options.tsx
"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";

interface Game {
  id: string;
  homeTeam: string;
  awayTeam: string;
}

interface BettingOptionsProps {
  game: Game;
}

export function BettingOptions({ game }: BettingOptionsProps) {
  return (
    <div className="space-y-6">
      {/* Moneyline Section */}
      <Card>
        <CardHeader>
          <CardTitle>Moneyline</CardTitle>
        </CardHeader>
        <CardContent className="grid grid-cols-2 gap-4">
          <Button variant="outline">{game.homeTeam} (—)</Button>
          <Button variant="outline">{game.awayTeam} (—)</Button>
        </CardContent>
      </Card>

      {/* Spread Section */}
      <Card>
        <CardHeader>
          <CardTitle>Spread</CardTitle>
        </CardHeader>
        <CardContent className="grid grid-cols-2 gap-4">
          <Button variant="outline">{game.homeTeam} (—)</Button>
          <Button variant="outline">{game.awayTeam} (—)</Button>
        </CardContent>
      </Card>

      {/* Total Section */}
      <Card>
        <CardHeader>
          <CardTitle>Total</CardTitle>
        </CardHeader>
        <CardContent className="grid grid-cols-2 gap-4">
          <Button variant="outline">Over —</Button>
          <Button variant="outline">Under —</Button>
        </CardContent>
      </Card>

      {/* Place Bet Section */}
      <Card>
        <CardHeader>
          <CardTitle>Place Bet</CardTitle>
        </CardHeader>
        <CardContent>
          <Button className="w-full" disabled>
            Place Bet
          </Button>
        </CardContent>
      </Card>
    </div>
  );
}

