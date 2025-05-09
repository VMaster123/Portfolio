export default function AboutPage() {
  return (
    <div className="container mx-auto max-w-4xl space-y-8 px-4 py-12 md:py-16">
      <div className="space-y-4">
        <h1 className="text-3xl font-bold tracking-tighter sm:text-4xl md:text-5xl">About Sports Betting</h1>
        <p className="text-muted-foreground md:text-xl">
          Your trusted platform for live sports betting and real-time game tracking.
        </p>
      </div>

      <div className="grid gap-8">
        <div className="space-y-4">
          <h2 className="text-2xl font-bold">Our Platform</h2>
          <p className="text-muted-foreground">
            We provide a seamless sports betting experience with live game updates, real-time odds, and secure betting
            options. Our platform is designed for both novice and experienced bettors.
          </p>
        </div>

        <div className="space-y-4">
          <h2 className="text-2xl font-bold">Features</h2>
          <ul className="list-inside list-disc space-y-2 text-muted-foreground">
            <li>Live game tracking with real-time score updates</li>
            <li>Comprehensive betting options including moneyline, spread, and totals</li>
            <li>Secure and fast betting platform</li>
            <li>Detailed game statistics and analysis</li>
          </ul>
        </div>

        <div className="space-y-4">
          <h2 className="text-2xl font-bold">Responsible Gaming</h2>
          <p className="text-muted-foreground">
            We are committed to promoting responsible gaming. We provide tools and resources to help our users make
            informed decisions and maintain control over their betting activities.
          </p>
        </div>
      </div>
    </div>
  )
}

