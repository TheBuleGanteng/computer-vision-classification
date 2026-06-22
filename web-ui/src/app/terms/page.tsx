import type { Metadata } from "next"

export const metadata: Metadata = {
  title: "Terms of Use — Neural Architecture Explorer",
  description: "Acceptable-use terms for the Neural Architecture Explorer demo.",
}

export default function TermsPage() {
  return (
    <div className="container mx-auto max-w-3xl px-4 sm:px-6 py-8 prose prose-sm dark:prose-invert">
      <h1 className="text-2xl font-bold mb-2">Terms of Use</h1>
      <p className="text-sm text-muted-foreground mb-6">Last updated: 22 June 2026</p>

      <p className="mb-4">
        This Neural Architecture Explorer is a free educational demo provided by{" "}
        <strong>Kebayoran Technologies</strong> on an &ldquo;as is&rdquo; basis,
        without warranties of any kind. By using it you agree to the following.
      </p>

      <h2 className="text-lg font-semibold mt-6 mb-2">Acceptable use</h2>
      <ul className="list-disc pl-6 mb-4 space-y-1">
        <li>Use the demo only for lawful, educational, and personal purposes.</li>
        <li>
          Do not attempt to overload, disrupt, probe, or abuse the optimization
          endpoint or any other part of the service, and do not use automated
          tooling to generate excessive load.
        </li>
        <li>
          Do not attempt to gain unauthorized access to the service, its
          infrastructure, or any data it processes.
        </li>
        <li>
          Do not use the service to host, distribute, or process illegal content.
        </li>
      </ul>

      <h2 className="text-lg font-semibold mt-6 mb-2">Availability</h2>
      <p className="mb-4">
        The demo trains models on shared infrastructure and may be rate-limited,
        changed, or taken offline at any time without notice. Results are for
        illustrative and educational purposes only.
      </p>

      <h2 className="text-lg font-semibold mt-6 mb-2">Contact</h2>
      <p className="mb-4">
        For questions about these terms, contact Kebayoran Technologies at{" "}
        <a
          href="https://www.kebayorantechnologies.com"
          target="_blank"
          rel="noopener noreferrer"
          className="text-blue-500 hover:text-blue-700 underline"
        >
          kebayorantechnologies.com
        </a>
        .
      </p>
    </div>
  )
}
