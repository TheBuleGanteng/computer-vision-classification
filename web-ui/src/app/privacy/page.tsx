import type { Metadata } from "next"

export const metadata: Metadata = {
  title: "Privacy Policy — Neural Architecture Explorer",
  description: "How the Neural Architecture Explorer demo handles data.",
}

export default function PrivacyPage() {
  return (
    <div className="container mx-auto max-w-3xl px-4 sm:px-6 py-8 prose prose-sm dark:prose-invert">
      <h1 className="text-2xl font-bold mb-2">Privacy Policy</h1>
      <p className="text-sm text-muted-foreground mb-6">Last updated: 22 June 2026</p>

      <p className="mb-4">
        This page is an interactive educational demo operated by{" "}
        <strong>Kebayoran Technologies</strong> that visualizes neural-network
        architectures produced by automated hyperparameter optimization. This
        policy explains what data the demo handles.
      </p>

      <h2 className="text-lg font-semibold mt-6 mb-2">No image uploads</h2>
      <p className="mb-4">
        The demo does <strong>not</strong> accept image uploads or use your
        camera. All model training is performed against a fixed set of
        pre-bundled, publicly available datasets (for example MNIST and CIFAR).
        You select a dataset and optimization settings; you do not submit any
        images of your own.
      </p>

      <h2 className="text-lg font-semibold mt-6 mb-2">What we process</h2>
      <p className="mb-4">
        When you start an optimization run, the settings you choose (dataset,
        number of trials, epochs, and scoring weights) are sent to our
        self-hosted backend, which builds and trains models and returns metrics
        and visualizations. The resulting artifacts (metrics, plots, model
        summaries) are written to ephemeral server-side storage tied to the run
        and are not published or linked to your identity. They are not retained
        as a permanent record and may be cleared at any time.
      </p>

      <h2 className="text-lg font-semibold mt-6 mb-2">Self-hosted inference</h2>
      <p className="mb-4">
        All model building, training, and inference is performed by{" "}
        <strong>self-hosted models</strong> on infrastructure operated by
        Kebayoran Technologies. No third-party inference, classification, or AI
        service receives your data.
      </p>

      <h2 className="text-lg font-semibold mt-6 mb-2">No third-party tracking</h2>
      <p className="mb-4">
        This demo uses no analytics, advertising, or third-party tracking SDKs,
        and sets no non-essential third-party cookies. Only first-party,
        session-scoped state required for the application to function is used.
      </p>

      <h2 className="text-lg font-semibold mt-6 mb-2">Contact</h2>
      <p className="mb-4">
        Questions about this policy can be directed to Kebayoran Technologies at{" "}
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
