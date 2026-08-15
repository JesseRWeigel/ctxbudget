# On measuring what a window holds

A context window is quoted in tokens, and almost nobody counts tokens. They count characters and
divide by four, which is a rule of thumb borrowed from English prose and applied to whatever
happens to be in the buffer. On ordinary paragraphs the rule is close enough that nobody notices.
On a minified bundle it is wildly optimistic, because the tokenizer cannot find familiar words to
merge and falls back to short pieces. On Japanese it is optimistic for the opposite reason, since
a single character occupies three bytes and the merge table has fewer of those sequences learned.

The consequence is not academic. A retrieval pipeline that packs twelve documents into a window
on the strength of a character estimate will silently drop the twelfth, and the model will answer
confidently from eleven. Nothing in the transcript records the loss. The answer looks the same
shape as a correct one, which is the property that makes this class of bug expensive.

There is a second arithmetic error underneath the first. The window is shared between what you
send and what comes back. A model advertising two hundred thousand tokens of context does not
offer two hundred thousand tokens of input, because the completion has to fit somewhere, and for
several hosted models the reply is separately capped well below the window. If you fill the
window to ninety-nine percent, the request may be accepted and the answer truncated mid-sentence.

Three habits fix most of this. Count with the tokenizer the model actually uses, or with an
approximation whose error you have measured rather than assumed. Subtract the reply budget before
you decide what fits. And when it does not fit, cut by something better than file size, because
the largest file in a set is often the one carrying the schema everything else refers to.
