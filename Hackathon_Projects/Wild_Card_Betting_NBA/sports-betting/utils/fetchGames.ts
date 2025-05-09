// utils/fetchGames.ts
import axios from "axios";

export async function fetchLiveGames() {
  try {
    const response = await axios.get(
      "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json"
    );
    const gamesList = response.data.scoreboard.games;

    // Transform the data to match your frontend structure
    const games = gamesList.map((game: any) => ({
      id: game.gameId,
      homeTeam: game.homeTeam.teamName,
      awayTeam: game.awayTeam.teamName,
      homeScore: game.homeTeam.score,
      awayScore: game.awayTeam.score,
      status: game.gameStatusText,
      quarter: game.period,
      timeLeft: game.gameClock,
    }));

    return games;
  } catch (error) {
    console.error("Error fetching live games:", error);
    return [];
  }
}

export async function fetchGameById(gameId: string) {
  try {
    const games = await fetchLiveGames();
    const game = games.find((game: any) => game.id === gameId);
    return game || null;
  } catch (error) {
    console.error("Error fetching game by ID:", error);
    return null;
  }
}

