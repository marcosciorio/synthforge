const FAQ_ITEMS: { q: string; a: React.ReactNode }[] = [
  {
    q: "Who are you?",
    a: (
      <>
        <p>
          I'm Marco Sciorio, a Master's student in Computer Engineering at the
          University of Naples Federico II. Since April 2026 I've been in London,
          at Queen Mary University of London - Centre for Digital Music,
          developing the research for my thesis.
        </p>
        <p>Here's where to find me if you'd like to know more:</p>
        <ul className="list-disc list-inside space-y-1">
          <li>
            Email:{" "}
            <a
              className="text-indigo-600 hover:underline break-all"
              href="mailto:ma.sciorio@studenti.unina.it"
            >
              ma.sciorio@studenti.unina.it
            </a>
          </li>
          <li>
            Instagram:{" "}
            <a
              className="text-indigo-600 hover:underline"
              href="https://www.instagram.com/marco.sciorio/"
              target="_blank"
              rel="noopener noreferrer"
            >
              @marco.sciorio
            </a>
          </li>
          <li>
            And if you'd like to listen to some of my music:{" "}
            <a
              className="text-indigo-600 hover:underline"
              href="https://open.spotify.com/intl-it/artist/44NCvHtfIwGc28uAhzLRcu?si=mDMo8NbqReaHwdvb8ier-w"
              target="_blank"
              rel="noopener noreferrer"
            >
              my Spotify
            </a>
          </li>
        </ul>
      </>
    ),
  },
  {
    q: "What are you working on?",
    a: (
      <>
        <p>
          I'm building an AI system that translates text descriptions of a sound
          into the parameters of a synthesizer — basically, an assistant that
          helps musicians and producers find the sound they have in mind just by
          describing it in words.
        </p>
        <p>
          To find out whether the system really works with the language real
          people use, I need your help: listening to a few sounds and describing
          them. Your descriptions let me test the model under realistic
          conditions, not just in the lab. It's a small contribution on your end
          (just a few minutes) but a very valuable one for the scientific quality
          of the project.
        </p>
      </>
    ),
  },
  {
    q: "There are already many text-to-audio models out there. How is your project different?",
    a: (
      <>
        <p>
          Existing models take a text prompt and produce an audio file directly —
          a "finished" waveform that you can only listen to as it is. If you don't
          like the result, all you can do is regenerate it and hope for something
          better.
        </p>
        <p>
          My system works in a fundamentally different way: starting from your
          text description, it doesn't generate an audio file — it finds the
          exact configuration of a synthesizer, i.e. the precise value of every
          knob, slider and switch. The result is a complete preset that you can
          load into the synth, play with your keyboard, and most importantly, edit
          freely: if the sound is almost right but you want more reverb, you turn
          a knob up; if the attack is too slow, you shorten it. You get full
          creative control.
        </p>
        <p>
          For a musician or producer, that difference is huge: instead of an
          opaque, unchangeable audio clip, you get a living, editable instrument
          that you can drop into your project exactly as you would with a preset
          you found by hand — without spending hours turning knobs blindly.
        </p>
      </>
    ),
  },
  {
    q: "How will my descriptions be used?",
    a: (
      <>
        <p>
          The descriptions you write are used exclusively to evaluate how well the
          model understands musical language. Here's the process:
        </p>
        <ul className="list-disc list-inside space-y-1">
          <li>You listen to a synthesizer sound and describe it in your own words.</li>
          <li>
            I take your description and feed it to the model, which generates a
            synthesizer configuration (a set of parameters).
          </li>
          <li>That configuration produces a new sound.</li>
          <li>
            I compare the original sound (the one you listened to) with the sound
            the model generated from your description.
          </li>
        </ul>
        <p>
          If the two sounds are similar, it means the model understood what you
          meant. If they're different, it tells me where the model struggles so I
          can improve it. Your descriptions are therefore a measurement tool for
          the quality of the system — and they're valuable precisely because they
          come from real people, not from an algorithm.
        </p>
      </>
    ),
  },
  {
    q: "How does the model work, briefly?",
    a: (
      <>
        <p>
          At the heart of the system is a generative model based on a technique
          called Conditional Flow Matching. In simple terms: the model starts from
          a random point and follows, step by step, a "map of directions" learned
          during training, until it arrives at the synthesizer configuration that
          matches the description you gave. Your description acts as a guide,
          steering the path toward the right sound.
        </p>
        <p>
          The link between words and sounds is made possible by a second
          component (CLAP), which places text and audio in the same shared space,
          so the system can translate a description into a precise direction to
          follow.
        </p>
      </>
    ),
  },
  {
    q: "Where do these sounds come from?",
    a: (
      <>
        <p>
          All the sounds you hear on this site come from TAL-NoiseMaker, a free,
          open-source software synthesizer widely used in music production. They
          are real, hand-crafted presets, factory presets and presets shared by
          actual users. Each one was designed on
          purpose by someone, covering a wide range of timbres: pads, basses,
          leads, arps, drums, sound effects, and more.
        </p>
        <p>
          Each clip is 4 seconds long, but the note (a C3) is only held down
          for the first 3 seconds — the last second is the natural tail/release
          of the sound after the key is released, not silence or a glitch.
        </p>
      </>
    ),
  },
];

export default function FaqSection() {
  return (
    <section className="border-t border-slate-200 bg-slate-50">
      <div className="max-w-5xl mx-auto w-full px-4 sm:px-6 md:px-10 py-8 md:py-12">
        <h2 className="text-lg font-semibold text-slate-800">❓ About this project</h2>
        <p className="text-sm text-slate-500 mt-1 mb-5">
          A few questions people often ask before describing sounds.
        </p>
        <div className="flex flex-col gap-2 max-w-3xl">
          {FAQ_ITEMS.map((item, i) => (
            <details
              key={i}
              className="rounded-xl border border-slate-200 bg-white open:bg-indigo-50/30"
            >
              <summary className="cursor-pointer select-none px-4 py-3.5 text-sm sm:text-base font-medium text-slate-800">
                {item.q}
              </summary>
              <div className="px-4 pb-4 text-sm text-slate-600 space-y-2">{item.a}</div>
            </details>
          ))}
        </div>
      </div>
    </section>
  );
}
